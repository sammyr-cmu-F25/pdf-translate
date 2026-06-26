import concurrent.futures
import logging
import re
import unicodedata
from enum import Enum
from string import Template
from typing import Dict

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, wait_fixed

from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
)

log = logging.getLogger(__name__)


def _char_is_cjk(ch: str) -> bool:
    """True for CJK text characters, broadly defined. Includes Kangxi radicals
    (U+2F00-U+2FDF) and CJK radicals supplement (U+2E80-U+2EFF) because some
    source PDFs encode real Chinese characters with those codepoints, plus CJK
    punctuation and full-width forms. Used to keep CJK body text from being
    misclassified as math subscripts/formulas (and thus left untranslated)."""
    if not ch:
        return False
    o = ord(ch)
    return (
        0x3400 <= o <= 0x9FFF        # CJK Unified Ideographs (+ Ext A)
        or 0xF900 <= o <= 0xFAFF     # CJK Compatibility Ideographs
        or 0x2E80 <= o <= 0x2FDF     # CJK Radicals Supplement + Kangxi Radicals
        or 0x3000 <= o <= 0x303F     # CJK Symbols and Punctuation
        or 0xFF00 <= o <= 0xFFEF     # Halfwidth/Fullwidth Forms (，。 etc.)
        or 0x20000 <= o <= 0x2A6DF   # CJK Unified Ideographs Ext B
    )


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # 重载替换 cropbox
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # 重载返回指令流
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # 重载设置 pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 重载返回指令流
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # 重载设置 cid 和 font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack 插入原字符编码
        item.font = font  # hack 插入原字符字体
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk, color=None):
        self.y: float = y  # 初始纵坐标
        self.x: float = x  # 初始横坐标
        self.x0: float = x0  # 左边界
        self.x1: float = x1  # 右边界
        self.y0: float = y0  # 上边界
        self.y1: float = y1  # 下边界
        self.size: float = size  # 字体大小
        self.brk: bool = brk  # 换行标记
        self.color = color  # PATCH#1 原文填充颜色 (graphicstate.ncolor)
        self.in_figure: bool = False  # PATCH#2 是否位于图表/表格区域(标签需更激进地缩放以塞进方框)
        self.region: int = -1  # PATCH#5 所属 layout 区域 id(用于同一表格内字号统一)


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        # PATCH#1b 颜色预言机：(pageid, x0, y0, x1, y1) -> 前景色 或 None
        # 由 high_level.translate_patch 用 PyMuPDF 解析的真实颜色注入，
        # 用于修正 pdfminer 把"白字+黑色阴影"误判为黑色的情况。
        self.color_oracle = None
        # PATCH#2 方框宽度预言机：(pageid,x0,y0,x1,y1)->真实方框宽度 或 None
        # 用于把图表标签精确缩放进其所在的流程图方框/表格单元格。
        self.box_oracle = None
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator, QwenMtTranslator,]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

    def receive_layout(self, ltpage: LTPage):
        # PATCH#2 图表/表格区域的 region id 集合（每行单独成段，保持锚定）
        try:
            import pdf2zh_patch as _pp
            _figure_region_ids = _pp.FIGURE_BOX_INDICES.get("value", set())
        except Exception:
            _figure_region_ids = set()
        # 段落
        sstk: list[str] = []            # 段落文字栈
        pstk: list[Paragraph] = []      # 段落属性栈
        vbkt: int = 0                   # 段落公式括号计数
        # 公式组
        vstk: list[LTChar] = []         # 公式符号组
        vlstk: list[LTLine] = []        # 公式线条组
        vfix: float = 0                 # 公式纵向偏移
        # 公式组栈
        var: list[list[LTChar]] = []    # 公式符号组栈
        varl: list[list[LTLine]] = []   # 公式线条组栈
        varf: list[float] = []          # 公式纵向偏移栈
        vlen: list[float] = []          # 公式宽度栈
        # 全局
        lstk: list[LTLine] = []         # 全局线条栈
        xt: LTChar = None               # 上一个字符
        xt_cls: int = -1                # 上一个字符所属段落，保证无论第一个字符属于哪个类别都可以触发新段落
        vmax: float = ltpage.width / 4  # 行内公式最大宽度
        ops: str = ""                   # 渲染结果

        def vflag(font: str, char: str):    # 匹配公式（和角标）字体
            if isinstance(font, bytes):     # 不一定能 decode，直接转 str
                try:
                    font = font.decode('utf-8')  # 尝试使用 UTF-8 解码
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # 字体名截断
            if re.match(r"\(cid:", char):
                return True
            # 基于字体名规则的判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # latex 字体
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # 基于字符集规则的判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                # PATCH: 常见算术/比较符号(+ = × ÷ ± < > 等)在普通文本/表格中是正文
                # (如 "12+"、"3×24"、"12-18%")，不应当作公式抽取，否则相邻数字/单位会被
                # 拆开单独翻译(如 "12 月" -> "December")。仅排除这些常见符号，真正的
                # 数学公式仍由字体名规则与 (cid:) 规则捕获。
                _common_math = set("+=×÷±<>~")
                if (
                    char
                    and char != " "                                     # 非空格
                    and char[0] not in _common_math                     # 常见算术符号视为正文
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 文字修饰符、数学符号、分隔符号
                        or ord(char[0]) in range(0x370, 0x400)          # 希腊字母
                    )
                ):
                    return True
            return False

        ############################################################
        # A. 原文档解析
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前字符在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                # 锚定文档中 bullet 的位置
                if child.get_text() == "•":
                    cls = 0
                # PATCH: 角标判定(规则2)不应作用于 CJK 表意文字。中文几乎没有数学角标，
                # 而一旦段落字号被装饰性大号标点(如放大的引号「」)抬高，正文 CJK 就会
                # 因 size < 段落size*0.79 被误判为角标/公式而漏译。
                _ctext = child.get_text()
                _is_cjk = bool(_ctext) and _char_is_cjk(_ctext[0])
                # 判定当前字符是否属于公式
                if (                                                                                        # 判定当前字符是否属于公式
                    cls == 0                                                                                # 1. 类别为保留区域
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79 and not _is_cjk)  # 2. 角标字体(CJK 文字除外)
                    or vflag(child.fontname, child.get_text())                                              # 3. 公式字体
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 垂直字体
                ):
                    cur_v = True
                # 判定括号组是否属于公式
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # 判定当前公式是否结束
                    not cur_v                                               # 1. 当前字符不属于公式
                    or cls != xt_cls                                        # 2. 当前字符与前一个字符不属于同一段落
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内换行，可能是一长串斜体的段落，也可能是段内分式换行，这里设个阈值进行区分
                    # 禁止纯公式（代码）段落换行，直到文字开始再重开文字段落，保证只存在两种情况
                    # A. 纯公式（代码）段落（锚定绝对位置）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. 文字开头段落（排版相对位置）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 因为 cls==xt_cls==0 一定有 sstk[-1]==""，所以这里不需要再判定 cls!=0
                ):
                    if vstk:
                        if (                                                # 根据公式右侧的文字修正公式的纵向偏移
                            not cur_v                                       # 1. 当前字符不属于公式
                            and cls == xt_cls                               # 2. 当前字符与前一个字符属于同一段落
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 当前字符在公式右侧
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 禁止纯公式段落（sstk[-1]=="{v*}"）的后续连接，但是要考虑新字符和后续字符的连接，所以这里修改的是上个字符的类别
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # PATCH#2 判断当前字符是否位于图表/表格区域。这类区域内的每个文本行
                # （流程图方框、表格单元格标签）应各自成段并锚定在原位，避免被合并成
                # 一长串重新排版而脱离原方框。
                in_figure = cls in _figure_region_ids
                # PATCH#2 图表区域内：垂直换行(下一行)或较大的水平跳变(跳到相邻方框)
                # 都应另起一段，使每个方框/单元格的标签各自独立锚定。
                fig_break = False
                if in_figure and xt is not None:
                    if child.x1 < xt.x0:                       # 垂直换行
                        fig_break = True
                    elif child.x0 > xt.x1 + 1.2 * child.size:  # 跳到相邻方框/单元格(列间隙)
                        # 列间隙通常 15-20pt，远大于词间空格(3-5pt)，用 1.2*字号 区分
                        fig_break = True

                # 当前字符不属于公式或当前字符是公式的第一个字符
                if not vstk:
                    if cls == xt_cls and not fig_break:
                        # 同一段落（且不是图表区域内的换行/跳格）
                        if child.x0 > xt.x1 + 1:    # 添加行内空格
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # 添加换行空格并标记原文段落存在换行
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:                           # 根据当前字符构建一个新的段落
                        sstk.append("")
                        # PATCH#1 记录原文字符的填充颜色，供新文档排版时复用
                        try:
                            _color = child.graphicstate.ncolor
                        except Exception:
                            _color = None
                        _para = Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False, color=_color)
                        _para.in_figure = in_figure  # PATCH#2 标记图表区域段落
                        _para.region = int(cls) if in_figure else -1  # PATCH#5 记录图表/表格区域 id
                        pstk.append(_para)
                if not cur_v:                                               # 文字入栈
                    # PATCH: 装饰性大号标点（如放大的引号「」）不应抬高段落字号，否则
                    # 后续正文会因 size < 段落size*0.79 被误判为角标/公式而漏译。
                    _ct = child.get_text()
                    _is_punct = bool(_ct) and unicodedata.category(_ct[0]) in (
                        "Ps", "Pe", "Pi", "Pf", "Po"  # 各类标点
                    )
                    if (                                                    # 根据当前字符修正段落属性
                        child.size > pstk[-1].size                          # 1. 当前字符比段落字体大
                        or len(sstk[-1].strip()) == 1                       # 2. 当前字符为段落第二个文字（考虑首字母放大的情况）
                    ) and _ct != " " and not _is_punct:                     # 3. 不是空格，也不是装饰性标点
                        pstk[-1].y -= child.size - pstk[-1].size            # 修正段落初始纵坐标，假设两个不同大小字符的上边界对齐
                        pstk[-1].size = child.size
                    sstk[-1] += _ct
                else:                                                       # 公式入栈
                    if (                                                    # 根据公式左侧的文字修正公式的纵向偏移
                        not vstk                                            # 1. 当前字符是公式的第一个字符
                        and cls == xt_cls                                   # 2. 当前字符与前一个字符属于同一段落
                        and child.x0 > xt.x0                                # 3. 前一个字符在公式左侧
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 更新段落边界，因为段落内换行之后可能是公式开头，所以要在外边处理
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 更新上一个字符
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 图表
                pass
            elif isinstance(child, LTLine):     # 线条
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前线条在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 公式线条
                    vlstk.append(child)
                else:                           # 全局线条
                    lstk.append(child)
            else:
                pass
        # 处理结尾
        if vstk:    # 公式出栈
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        log.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # 计算公式宽度
            l = max([vch.x1 for vch in v]) - v[0].x0
            log.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. 段落翻译
        log.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # 多线程翻译
            if not s.strip() or re.match(r"^\{v\d+\}$", s):  # 空白和公式不翻译
                return s
            try:
                new = self.translator.translate(s)
                return new
            except BaseException as e:
                if log.isEnabledFor(logging.DEBUG):
                    log.exception(e)
                else:
                    log.exception(e, exc_info=False)
                raise e
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread
        ) as executor:
            news = list(executor.map(worker, sstk))

        ############################################################
        # C. 新文档排版
        def raw_string(fcur: str, cstk: str):  # 编码字符串
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # 判断编码长度
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # 根据目标语言获取默认行距
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # 小语种默认1.1
        _x, _y = 0, 0
        ops_list = []

        def gen_op_color(color):  # PATCH#1 生成 PDF 填充颜色操作符
            # pdfminer 的 ncolor 可能是 float(灰度) / 3元组(RGB) / 4元组(CMYK) / None
            try:
                if color is None:
                    return ""
                if isinstance(color, (int, float)):
                    g = max(0.0, min(1.0, float(color)))
                    return f"{g:f} g "
                if isinstance(color, (tuple, list)):
                    vals = [max(0.0, min(1.0, float(v))) for v in color]
                    if len(vals) == 1:
                        return f"{vals[0]:f} g "
                    if len(vals) == 3:
                        return f"{vals[0]:f} {vals[1]:f} {vals[2]:f} rg "
                    if len(vals) == 4:
                        return f"{vals[0]:f} {vals[1]:f} {vals[2]:f} {vals[3]:f} k "
            except Exception:
                return ""
            return ""

        def gen_op_txt(font, size, x, y, rtxt, color=None):
            return f"{gen_op_color(color)}/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        # 通用文字宽度测量（供缩放与单词换行预计算复用）
        def _measure_width_g(s: str, sz: float) -> float:
            w = 0.0
            p = 0
            while p < len(s):
                mm = re.match(r"\{\s*v([\d\s]+)\}", s[p:], re.IGNORECASE)
                if mm:
                    p += len(mm.group(0))
                    try:
                        w += vlen[int(mm.group(1).replace(" ", ""))]
                    except Exception:
                        pass
                    continue
                c = s[p]; p += 1
                f_ = None
                try:
                    if self.fontmap["tiro"].to_unichr(ord(c)) == c:
                        f_ = "tiro"
                except Exception:
                    pass
                if f_ is None:
                    f_ = self.noto_name
                if f_ == self.noto_name:
                    w += self.noto.char_lengths(c, sz)[0]
                else:
                    w += self.fontmap[f_].char_width(ord(c)) * sz
            return w

        # PATCH#5 预计算每段缩放后字号，并按 layout 区域(表格/图表)取最小值统一，
        # 避免同一表格内各单元格字号忽大忽小。
        def _shrunk_size(idx):
            p = pstk[idx]
            sz = p.size
            bw = p.x1 - p.x0
            if p.in_figure and self.box_oracle is not None:
                try:
                    rw = self.box_oracle(ltpage.pageid, p.x0, p.y0, p.x1, p.y1)
                except Exception:
                    rw = None
                if rw and rw > 1:
                    bw = rw * 0.92
            # 只对图表/表格内文字(固定方框、无法换行)做缩小。普通正文/标题宁可换行，
            # 不缩小——否则标题会被缩得比正文还小(标题原文常被拆成很窄的几段)。
            if p.in_figure and bw > 1:
                nat = _measure_width_g(news[idx], sz)
                if nat > bw + 0.1 * sz:
                    sz = max(sz * 0.25, sz * (bw / nat))
            return sz

        _region_min = {}
        for _i in range(len(news)):
            _r = pstk[_i].region
            if _r >= 0:
                _s = _shrunk_size(_i)
                _region_min[_r] = min(_region_min.get(_r, _s), _s)

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # 段落初始横坐标
            y: float = pstk[id].y                       # 段落初始纵坐标
            x0: float = pstk[id].x0                     # 段落左边界
            x1: float = pstk[id].x1                     # 段落右边界
            height: float = pstk[id].y1 - pstk[id].y0   # 段落高度
            size: float = pstk[id].size                 # 段落字体大小
            brk: bool = pstk[id].brk                    # 段落换行标记
            color = pstk[id].color                      # PATCH#1 段落原文颜色

            # PATCH#1b 当 pdfminer 报告黑色/无色时，向颜色预言机(PyMuPDF)询问该区域的真实前景色。
            # 处理"白字 + 黑色阴影"叠绘被误判为黑色的情况。
            def _is_blackish(c):
                if c is None:
                    return True
                if isinstance(c, (int, float)):
                    return float(c) <= 0.01
                if isinstance(c, (tuple, list)):
                    return all(float(v) <= 0.01 for v in c)
                return False

            # 注意：figure(图表)内部的文字 receive_layout 会以局部坐标单独调用一次，
            # 这些坐标与整页 PyMuPDF span 对不齐，容易误命中别处的彩色文字。
            # 图表内文字 pdfminer 的颜色本就正确（通常为黑），因此只在整页层面咨询预言机。
            if (
                self.color_oracle is not None
                and _is_blackish(color)
                and not isinstance(ltpage, LTFigure)
            ):
                try:
                    oracle_color = self.color_oracle(ltpage.pageid, pstk[id].x0, pstk[id].y0, pstk[id].x1, pstk[id].y1)
                except Exception:
                    oracle_color = None
                if oracle_color is not None:
                    color = oracle_color

            # PATCH#3 自适应缩小字号：当译文在单行内放不下时（标题/表格单元格等不换行的段落），
            # 按比例缩小字号以塞进原始宽度，避免溢出单元格/重叠。
            def _measure_width(s: str, sz: float) -> float:
                w = 0.0
                p = 0
                while p < len(s):
                    m = re.match(r"\{\s*v([\d\s]+)\}", s[p:], re.IGNORECASE)
                    if m:  # 公式按其已知宽度计（与字号无关）
                        p += len(m.group(0))
                        try:
                            w += vlen[int(m.group(1).replace(" ", ""))]
                        except Exception:
                            pass
                        continue
                    c = s[p]; p += 1
                    f_ = None
                    try:
                        if self.fontmap["tiro"].to_unichr(ord(c)) == c:
                            f_ = "tiro"
                    except Exception:
                        pass
                    if f_ is None:
                        f_ = self.noto_name
                    if f_ == self.noto_name:
                        w += self.noto.char_lengths(c, sz)[0]
                    else:
                        w += self.fontmap[f_].char_width(ord(c)) * sz
                return w

            in_fig = pstk[id].in_figure
            # PATCH#5 表格/图表区域：使用区域内统一(最小)字号，保证同表格字号一致。
            # 其它段落：按自身缩放。
            _reg = pstk[id].region
            if _reg >= 0 and _reg in _region_min:
                size = _region_min[_reg]
            else:
                size = _shrunk_size(id)

            # PATCH#4 单词级换行：对含空格的(拉丁)译文，预先在空格处计算换行点，
            # 使其按单词边界换行，而不是在单词中间硬切（如 "p / ositive"）。
            # - brk 段落(正文)：按其原始文本宽度换行。
            # - 非 brk 的非图表段落(如标题)：原文常被拆成很窄的几段(x1-x0 极小)，
            #   不能按该宽度换行，否则每个词都换行；改用到页面右边距的可用宽度。
            wrap_breaks = set()
            _wrap_avail = x1 - x0
            if (not brk) and (not in_fig):
                page_right = ltpage.width - 56.0  # 估计右边距
                if page_right - x0 > _wrap_avail:
                    _wrap_avail = page_right - x0
            if _wrap_avail > 1 and " " in new.strip() and (brk or not in_fig):
                avail = _wrap_avail
                line_w = 0.0
                last_space_ptr = None
                seg_w_since_space = 0.0  # 自上个空格以来的宽度
                i = 0
                while i < len(new):
                    # {vN} 公式记号作为一个不可断开的整体，按其已知宽度计
                    vm = re.match(r"\{\s*v([\d\s]+)\}", new[i:], re.IGNORECASE)
                    if vm:
                        try:
                            cw = vlen[int(vm.group(1).replace(" ", ""))]
                        except Exception:
                            cw = 0.0
                        if line_w + cw > avail and last_space_ptr is not None:
                            wrap_breaks.add(last_space_ptr)
                            line_w = seg_w_since_space + cw
                            last_space_ptr = None
                        else:
                            line_w += cw
                        seg_w_since_space += cw
                        i += len(vm.group(0))
                        continue
                    cc = new[i]
                    f2 = None
                    try:
                        if self.fontmap["tiro"].to_unichr(ord(cc)) == cc:
                            f2 = "tiro"
                    except Exception:
                        pass
                    if f2 is None:
                        f2 = self.noto_name
                    if f2 == self.noto_name:
                        cw = self.noto.char_lengths(cc, size)[0]
                    else:
                        cw = self.fontmap[f2].char_width(ord(cc)) * size
                    if cc == " ":
                        last_space_ptr = i
                        seg_w_since_space = 0.0
                    else:
                        seg_w_since_space += cw
                    if line_w + cw > avail and last_space_ptr is not None:
                        # 在最近的空格处换行
                        wrap_breaks.add(last_space_ptr)
                        line_w = seg_w_since_space  # 新行从该单词开始
                        last_space_ptr = None
                    else:
                        line_w += cw
                    i += 1

            cstk: str = ""                              # 当前文字栈
            fcur: str = None                            # 当前字体 ID
            lidx = 0                                    # 记录换行次数
            tx = x
            fcur_ = fcur
            ptr = 0
            log.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ops_vals: list[dict] = []

            while ptr < len(new):
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # 匹配 {vn} 公式标记
                mod = 0  # 文字修饰符
                if vy_regex:  # 加载公式
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # 翻译器可能会自动补个越界的公式标记
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # 文字修饰符
                        mod = var[vid][-1].width
                else:  # 加载文字
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"  # 默认拉丁字体
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 默认非拉丁字体
                    if fcur_ == self.noto_name: # FIXME: change to CONST
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                # PATCH#4 单词级换行模式：在预计算的空格位置换行；否则沿用按宽度换行
                word_wrap = bool(wrap_breaks)
                ptr_at_char = ptr - 1 if not vy_regex else ptr  # 当前字符在 new 中的索引
                hit_word_break = word_wrap and (ptr_at_char in wrap_breaks)
                hit_edge = (not word_wrap) and (x + adv > x1 + 0.1 * size)

                if (                                # 输出文字缓冲区
                    fcur_ != fcur                   # 1. 字体更新
                    or vy_regex                     # 2. 插入公式
                    or hit_edge                     # 3. 到达右边界
                    or hit_word_break               # 4. 单词边界换行点
                ):
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx
                        })
                        cstk = ""
                # 换行：brk 段落到右边界换行；任何段落命中预计算的单词换行点都换行
                if (brk and hit_edge) or hit_word_break:
                    x = x0
                    lidx += 1
                if vy_regex:  # 插入公式
                    fix = 0
                    if fcur is not None:  # 段落内公式修正纵向偏移
                        fix = varf[vid]
                    for vch in var[vid]:  # 排版公式字符
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx,
                            "is_formula": True,  # PATCH#1 公式字符保留原色
                        })
                        if log.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # 排版公式线条
                        if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # 插入文字缓冲区
                    if not cstk:  # 单行开头
                        tx = x
                        if x == x0 and ch == " ":  # 消除段落换行空格
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # 文字修饰符
                fcur = fcur_
                x += adv
                if log.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 处理结尾
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    # PATCH#1 公式字符保留原色(color=None)，译文正文使用段落原文颜色
                    op_color = None if vals.get("is_formula") else color
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"], color=op_color))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        for l in lstk:  # 排版全局线条
            if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ops = f"BT {''.join(ops_list)}ET "
        return ops


class OpType(Enum):
    TEXT = "text"
    LINE = "line"
