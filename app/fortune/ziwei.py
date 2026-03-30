"""
紫微斗數排盤模組
使用 iztro-py 排出命盤，翻譯成繁體中文，供 LLM 解讀
"""
import logging
from typing import Optional

from iztro_py import astro
from iztro_py.i18n.locales.zh_TW import translations

# ── 翻譯工具 ──

_PALACE_MAP = translations["palaces"]
_MAJOR_MAP = translations["stars"]["major"]
_MINOR_MAP = translations["stars"]["minor"]
_STEM_MAP = translations["heavenlyStem"]
_BRANCH_MAP = translations["earthlyBranch"]

# 雜曜翻譯（iztro-py 沒完整提供，手動補）
_ADJ_MAP = {
    "huagai": "華蓋", "xianchi": "咸池", "guchen": "孤辰", "guasu": "寡宿",
    "tiancai": "天才", "tianshou": "天壽", "hongluan": "紅鸞", "tianxi": "天喜",
    "tianxing": "天刑", "tianyao": "天姚", "jieshen": "解神", "yinsha": "陰煞",
    "tianguan": "天官", "tianfuAdj": "天福", "tianku": "天哭", "tianxu": "天虛",
    "longchi": "龍池", "fengge": "鳳閣", "feilian": "飛廉", "posui": "破碎",
    "tianchu": "天廚", "santai": "三台", "bazuo": "八座", "enguang": "恩光",
    "tiangui": "天貴", "taifu": "台輔", "fenggao": "封誥", "tianwu": "天巫",
    "tianyue": "天月", "tiande": "天德", "yuede": "月德", "tiankong": "天空",
    "xunkong": "旬空", "jielu": "截路", "kongwang": "空亡", "longde": "龍德",
    "jiekong": "截空", "jieshaAdj": "劫煞", "dahaoAdj": "大耗", "tianshi": "天使",
    "tianshang": "天傷", "nianjie": "年解",
}

# 四化翻譯
_MUTAGEN_MAP = {"禄": "化祿", "权": "化權", "科": "化科", "忌": "化忌",
                "祿": "化祿", "權": "化權", "科": "化科"}


def _star_name(star) -> str:
    """將 star 物件翻譯為中文名稱"""
    name = star.name
    cn = _MAJOR_MAP.get(name) or _MINOR_MAP.get(name) or _ADJ_MAP.get(name) or name
    parts = [cn]
    if star.brightness:
        parts.append(f"({star.brightness})")
    if star.mutagen:
        parts.append(_MUTAGEN_MAP.get(star.mutagen, f"({star.mutagen})"))
    return "".join(parts)


def _palace_name(name: str) -> str:
    return _PALACE_MAP.get(name, name)


def _branch_name(branch: str) -> str:
    return _BRANCH_MAP.get(branch, branch)


# 時辰對照（小時 → 時辰索引）
_HOUR_TO_INDEX = {
    0: 0, 1: 0,      # 子 (23:00-01:00)
    2: 1, 3: 1,      # 丑
    4: 2, 5: 2,      # 寅
    6: 3, 7: 3,      # 卯
    8: 4, 9: 4,      # 辰
    10: 5, 11: 5,    # 巳
    12: 6, 13: 6,    # 午
    14: 7, 15: 7,    # 未
    16: 8, 17: 8,    # 申
    18: 9, 19: 9,    # 酉
    20: 10, 21: 10,  # 戌
    22: 11, 23: 11,  # 亥
}

_SHICHEN_NAMES = ["子時", "丑時", "寅時", "卯時", "辰時", "巳時",
                  "午時", "未時", "申時", "酉時", "戌時", "亥時"]


def generate_chart(
    solar_date: str,
    birth_hour: int,
    gender: str,
) -> Optional[str]:
    """
    排出紫微命盤並回傳格式化的中文文字。

    Args:
        solar_date: 國曆生日，格式 "YYYY-M-D" 或 "YYYY-MM-DD"
        birth_hour: 出生時辰索引 (0=子, 1=丑, ..., 11=亥) 或小時 (0-23)
        gender: "男" 或 "女"

    Returns:
        格式化命盤文字 (str)，供 LLM 做為 context 解讀
    """
    try:
        # 如果傳入的是小時 (0-23)，轉為時辰索引 (0-11)
        if birth_hour > 11:
            birth_hour = _HOUR_TO_INDEX.get(birth_hour, 0)

        result = astro.by_solar(solar_date, birth_hour, gender, language='zh-TW')

        lines = []
        lines.append("═══ 紫微斗數命盤 ═══")
        lines.append(f"陽曆：{result.solar_date}")
        lines.append(f"農曆：{result.lunar_date}")
        lines.append(f"干支：{result.chinese_date}")
        lines.append(f"時辰：{_SHICHEN_NAMES[birth_hour]}")
        lines.append(f"性別：{gender}")
        lines.append(f"星座：{result.sign}")
        lines.append(f"生肖：{result.zodiac}")
        lines.append(f"五行局：{result.five_elements_class}")
        lines.append(f"命主星：{_MAJOR_MAP.get(result.soul, result.soul)}")
        lines.append(f"身主星：{_MAJOR_MAP.get(result.body, result.body)}")
        lines.append(f"命宮地支：{_branch_name(result.earthly_branch_of_soul_palace)}")
        lines.append(f"身宮地支：{_branch_name(result.earthly_branch_of_body_palace)}")
        lines.append("")

        # 十二宮位
        lines.append("═══ 十二宮位 ═══")
        for p in result.palaces:
            pname = _palace_name(p.name)
            branch = _branch_name(p.earthly_branch)
            body_mark = " 【身宮】" if p.is_body_palace else ""
            lines.append(f"\n▸ {pname}（{branch}）{body_mark}")

            majors = [_star_name(s) for s in p.major_stars]
            minors = [_star_name(s) for s in p.minor_stars]
            adjs = [_star_name(s) for s in p.adjective_stars]

            if majors:
                lines.append(f"  主星：{'、'.join(majors)}")
            else:
                lines.append("  主星：（空宮，借對宮星曜）")
            if minors:
                lines.append(f"  輔星：{'、'.join(minors)}")
            if adjs:
                lines.append(f"  雜曜：{'、'.join(adjs)}")

            lines.append(f"  長生12神：{p.changsheng12}")

        return "\n".join(lines)

    except Exception as e:
        logging.error(f"[ZiWei] Chart generation failed: {e}")
        return None


def parse_shichen(text: str) -> Optional[int]:
    """
    從使用者輸入解析時辰。
    支援：「子時」「凌晨1點」「早上8點」「下午3點」「15:00」「不知道」等
    回傳 0-11 的時辰索引，無法解析回傳 None
    """
    text = text.strip().replace("：", ":").replace("點", "").replace("時", "")

    # 直接匹配時辰名
    shichen_map = {
        "子": 0, "丑": 1, "寅": 2, "卯": 3, "辰": 4, "巳": 5,
        "午": 6, "未": 7, "申": 8, "酉": 9, "戌": 10, "亥": 11,
    }
    for k, v in shichen_map.items():
        if k in text:
            return v

    # 嘗試解析數字小時
    import re
    hour_match = re.search(r'(\d{1,2})', text)
    if hour_match:
        hour = int(hour_match.group(1))
        # 處理上午下午
        if ("下午" in text or "晚上" in text or "PM" in text.upper()) and hour < 12:
            hour += 12
        if 0 <= hour <= 23:
            return _HOUR_TO_INDEX.get(hour, 0)

    return None
