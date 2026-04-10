"""
Google Calendar 整合：查詢空檔 + 建立預約事件

前置作業（老闆做一次）：
1. 進 Google Calendar → 設定 → 把日曆共用給服務帳號（編輯者權限）
   服務帳號 email: bothelper-sheets@bothelper-489007.iam.gserviceaccount.com
2. 在後台設定頁填入 Calendar ID（在日曆設定中可以找到）
"""

import logging
from datetime import datetime, timedelta, timezone

from googleapiclient.discovery import build
from app.sheets.client import _get_credentials  # 重用同一組 Google 憑證


def _get_service():
    creds = _get_credentials()
    return build("calendar", "v3", credentials=creds)


def get_available_slots(
    calendar_id: str,
    date_str: str,
    duration_minutes: int,
    business_hours: dict
) -> list:
    """
    查詢指定日期的可預約時段（排除已有事件的時間）

    Args:
        calendar_id:      Google Calendar ID
        date_str:         日期字串 YYYY-MM-DD
        duration_minutes: 每次預約時長（分鐘）
        business_hours:   {"start": "09:00", "end": "18:00", "weekdays": [1,2,3,4,5]}
                          weekdays: 1=週一, 7=週日

    Returns:
        可預約時段列表，例如 ["09:00", "10:00", "14:00"]
    """
    try:
        service = _get_service()
        tz_tw = timezone(timedelta(hours=8))

        date = datetime.strptime(date_str, "%Y-%m-%d")
        weekday = date.isoweekday()  # 1=週一, 7=週日

        # 確認是否為上班日
        weekdays = business_hours.get("weekdays", [1, 2, 3, 4, 5])
        if weekday not in weekdays:
            return []

        # 解析上班時間
        start_h, start_m = map(int, business_hours.get("start", "09:00").split(":"))
        end_h, end_m = map(int, business_hours.get("end", "18:00").split(":"))

        day_start = datetime(date.year, date.month, date.day, start_h, start_m, tzinfo=tz_tw)
        day_end   = datetime(date.year, date.month, date.day, end_h,   end_m,   tzinfo=tz_tw)

        # 取得當天已有的事件（忙碌時段）
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        busy: list[tuple] = []
        for ev in result.get("items", []):
            ev_start = ev["start"].get("dateTime", "")
            ev_end   = ev["end"].get("dateTime",   "")
            if ev_start and ev_end:
                bs = datetime.fromisoformat(ev_start)
                be = datetime.fromisoformat(ev_end)
                if bs.tzinfo is None:
                    bs = bs.replace(tzinfo=tz_tw)
                if be.tzinfo is None:
                    be = be.replace(tzinfo=tz_tw)
                busy.append((bs, be))

        # 產生所有可能時段，排除忙碌的
        available = []
        cur = day_start
        while cur + timedelta(minutes=duration_minutes) <= day_end:
            slot_end = cur + timedelta(minutes=duration_minutes)
            is_free = all(slot_end <= bs or cur >= be for bs, be in busy)
            if is_free:
                available.append(cur.strftime("%H:%M"))
            cur += timedelta(minutes=duration_minutes)

        return available

    except Exception as e:
        logging.error(f"[Calendar] get_available_slots error: {e}")
        raise


def create_booking(
    calendar_id: str,
    title: str,
    date_str: str,
    time_str: str,
    duration_minutes: int,
    description: str = ""
) -> dict:
    """
    在 Google Calendar 建立預約事件

    Returns:
        {"event_id": str, "event_link": str}
    """
    try:
        service = _get_service()
        tz_tw = timezone(timedelta(hours=8))

        h, m = map(int, time_str.split(":"))
        date  = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime(date.year, date.month, date.day, h, m, tzinfo=tz_tw)
        end   = start + timedelta(minutes=duration_minutes)

        event = {
            "summary":     title,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": "Asia/Taipei"},
            "end":   {"dateTime": end.isoformat(),   "timeZone": "Asia/Taipei"},
        }

        created = service.events().insert(calendarId=calendar_id, body=event).execute()
        logging.info(f"[Calendar] Created event: {created.get('id')} at {date_str} {time_str}")
        return {
            "event_id":   created.get("id", ""),
            "event_link": created.get("htmlLink", ""),
        }

    except Exception as e:
        logging.error(f"[Calendar] create_booking error: {e}")
        raise
