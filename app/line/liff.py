"""
LIFF 快速填表（目前僅 697 這台 bot 使用，靠 bots.liff_intake_enabled 開關）。

流程：
  歡迎語附一張 Flex，按鈕開 LIFF 表單 → 客戶填 姓名/電話/地區/需求金額 → 送出
  → 後端 /liff/intake/submit 合併進該 session 的 collected_data → 接回問答。

本模組只負責「組 Flex」與「表單 HTML」，不碰其他 bot 的流程。
"""
from urllib.parse import quote

# 表單收集的欄位（鍵名務必對齊 697 prompt 的固定鍵名，卡片/webhook 才對得上）。
# 前四項必填，LINE ID 選填（很多人沒設，沒有就留空、靠手機聯繫）。
INTAKE_FIELDS = ["姓名", "電話", "地區", "需求金額", "LINE ID"]


def build_intake_flex(bot_id: str, session_id: str, liff_id: str) -> dict:
    """組一張帶『快速填資料』按鈕的 Flex 訊息（按鈕開 LIFF，網址帶 sid/bot）。"""
    url = f"https://liff.line.me/{liff_id}?sid={quote(session_id)}&bot={quote(bot_id)}"
    return {
        "type": "flex",
        "altText": "📝 點我 30 秒快速填寫資料，加快辦理",
        "contents": {
            "type": "bubble",
            "body": {
                "type": "box",
                "layout": "vertical",
                "spacing": "sm",
                "contents": [
                    {"type": "text", "text": "📝 30 秒快速填資料",
                     "weight": "bold", "size": "lg"},
                    {"type": "text",
                     "text": "填好基本資料，專員能更快幫您辦理。不想填也沒關係，直接跟我聊也可以喔！",
                     "size": "sm", "color": "#666666", "wrap": True},
                ],
            },
            "footer": {
                "type": "box",
                "layout": "vertical",
                "contents": [{
                    "type": "button",
                    "style": "primary",
                    "color": "#06C755",
                    "action": {"type": "uri", "label": "點我填寫", "uri": url},
                }],
            },
        },
    }


# 表單頁 HTML（自包含；%%PLACEHOLDER%% 由後端替換後回傳）
INTAKE_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>快速填資料</title>
<script src="https://static.line-scdn.net/liff/edge/2/sdk.js"></script>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,"Noto Sans TC",sans-serif; background:#f6f7f9; color:#1a1a1a; }
  .wrap { max-width:460px; margin:0 auto; padding:20px 16px 40px; }
  h1 { font-size:20px; margin:8px 0 4px; }
  p.sub { color:#666; font-size:13px; margin:0 0 20px; }
  label { display:block; font-size:14px; font-weight:600; margin:16px 0 6px; }
  .req { color:#e3382d; font-size:13px; }
  input { width:100%; padding:12px 14px; font-size:16px; border:1px solid #d0d4d9; border-radius:10px; background:#fff; }
  input:focus { outline:none; border-color:#06C755; }
  .hint { color:#999; font-size:12px; margin-top:4px; }
  button { width:100%; margin-top:24px; padding:14px; font-size:16px; font-weight:700; color:#fff; background:#06C755; border:none; border-radius:12px; }
  button:disabled { background:#9ad7b4; }
  .err { color:#e3382d; font-size:13px; margin-top:14px; min-height:18px; }
  .done { text-align:center; padding:48px 16px; }
  .done .big { font-size:40px; }
  #form.hide, #done.hide { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <div id="form">
    <h1>快速填資料</h1>
    <p class="sub">填好基本資料，專員能更快為您辦理（約 30 秒）</p>

    <label>姓名（本名）<span class="req"> *</span></label>
    <input id="f_name" type="text" autocomplete="name" placeholder="請填真實姓名">

    <label>手機號碼<span class="req"> *</span></label>
    <input id="f_phone" type="tel" inputmode="numeric" autocomplete="tel" placeholder="09xxxxxxxx">

    <label>方便辦理的地區<span class="req"> *</span></label>
    <input id="f_area" type="text" placeholder="例：台中市北區">
    <div class="hint">縣市＋行政區即可，不用填完整門牌</div>

    <label>需求金額<span class="req"> *</span></label>
    <input id="f_amount" type="text" inputmode="numeric" placeholder="例：100000">
    <div class="hint">直接填完整數字（元）</div>

    <label>LINE ID<span style="color:#999;font-weight:400"> （沒有可留空）</span></label>
    <input id="f_lineid" type="text" placeholder="有的話填，方便專員加您好友">

    <div class="err" id="err"></div>
    <button id="submit">送出</button>
  </div>

  <div id="done" class="hide">
    <div class="done">
      <div class="big">✅</div>
      <h1>收到了，謝謝您！</h1>
      <p class="sub">資料已送出，請回到聊天室，專員會盡快與您聯繫。</p>
    </div>
  </div>
</div>

<script>
var LIFF_ID = "%%LIFF_ID%%";
var qs = new URLSearchParams(location.search);
var SID = qs.get("sid") || "";
var BOT = qs.get("bot") || "";
var $ = function(id){ return document.getElementById(id); };

function showErr(m){ $("err").textContent = m || ""; }

// 不自動帶入 LINE 顯示名稱（常是綽號，697 要的是本名，留空讓客戶自己填）
liff.init({ liffId: LIFF_ID }).catch(function(e){ showErr("載入失敗，請關掉重開：" + e); });

$("submit").addEventListener("click", function(){
  var name = $("f_name").value.trim();
  var phone = $("f_phone").value.replace(/\\D/g, "");
  var area = $("f_area").value.trim();
  var amount = $("f_amount").value.trim();
  var lineid = $("f_lineid").value.trim();
  if (!name) return showErr("請填姓名");
  if (!/^09\\d{8}$/.test(phone)) return showErr("手機格式怪怪的，請填 09 開頭共 10 碼");
  if (!area) return showErr("請填方便辦理的地區");
  if (!amount) return showErr("請填需求金額");
  showErr("");
  $("submit").disabled = true;
  $("submit").textContent = "送出中…";

  var idToken = "";
  try { idToken = liff.getIDToken() || ""; } catch (e) {}

  fetch("/liff/intake/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sid: SID, bot: BOT, idToken: idToken,
      data: { "姓名": name, "電話": phone, "地區": area, "需求金額": amount, "LINE ID": lineid }
    })
  }).then(function(r){ return r.json().then(function(j){ return { ok: r.ok, j: j }; }); })
    .then(function(res){
      if (!res.ok || !res.j.ok) throw new Error((res.j && res.j.error) || "送出失敗");
      $("form").classList.add("hide");
      $("done").classList.remove("hide");
      setTimeout(function(){ try { liff.closeWindow(); } catch(e){} }, 1500);
    })
    .catch(function(e){
      $("submit").disabled = false;
      $("submit").textContent = "送出";
      showErr(String(e.message || e));
    });
});
</script>
</body>
</html>
"""


def render_intake_html(liff_id: str) -> str:
    return INTAKE_HTML.replace("%%LIFF_ID%%", liff_id or "")
