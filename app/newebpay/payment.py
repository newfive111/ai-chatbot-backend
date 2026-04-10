"""
藍新金流 (NewebPay) 付款工具
- AES-256-CBC 加密 TradeInfo
- SHA256 產生 TradeSha
- 建立付款參數
- 解密 Webhook 通知
"""
import hashlib
import time
import urllib.parse
import binascii
import logging

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def aes_encrypt(data: str, key: str, iv: str) -> str:
    """AES-256-CBC 加密，回傳 hex string"""
    key_b = key.encode("utf-8")
    iv_b  = iv.encode("utf-8")
    cipher = AES.new(key_b, AES.MODE_CBC, iv_b)
    encrypted = cipher.encrypt(pad(data.encode("utf-8"), AES.block_size))
    return binascii.hexlify(encrypted).decode("utf-8")


def aes_decrypt(hex_data: str, key: str, iv: str) -> str:
    """AES-256-CBC 解密"""
    key_b      = key.encode("utf-8")
    iv_b       = iv.encode("utf-8")
    enc_bytes  = binascii.unhexlify(hex_data)
    cipher     = AES.new(key_b, AES.MODE_CBC, iv_b)
    decrypted  = unpad(cipher.decrypt(enc_bytes), AES.block_size)
    return decrypted.decode("utf-8")


def sha256_mac(trade_info: str, key: str, iv: str) -> str:
    """產生 TradeSha：HashKey={key}&{trade_info}&HashIV={iv} → SHA256 大寫"""
    raw = f"HashKey={key}&{trade_info}&HashIV={iv}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def build_checkout_params(
    merchant_id: str,
    hash_key: str,
    hash_iv: str,
    order_no: str,
    amount: int,
    item_desc: str,
    email: str,
    return_url: str,
    notify_url: str,
    sandbox: bool = True,
) -> dict:
    """
    建立藍新 MPG 付款參數，回傳給前端使用。
    前端需要用 form POST 送到 gateway_url。
    """
    params = {
        "MerchantID":   merchant_id,
        "RespondType":  "JSON",
        "TimeStamp":    str(int(time.time())),
        "Version":      "2.0",
        "MerchantOrderNo": order_no,
        "Amt":          str(amount),
        "ItemDesc":     item_desc[:50],
        "TradeLimit":   "900",
        "ReturnURL":    return_url,
        "NotifyURL":    notify_url,
        "Email":        email,
        "EmailModify":  "0",
        "CREDIT":       "1",
    }

    trade_info_str = urllib.parse.urlencode(params)
    trade_info     = aes_encrypt(trade_info_str, hash_key, hash_iv)
    trade_sha      = sha256_mac(trade_info, hash_key, hash_iv)

    if sandbox:
        gateway_url = "https://ccore.newebpay.com/MPG/mpg_gateway"
    else:
        gateway_url = "https://core.newebpay.com/MPG/mpg_gateway"

    return {
        "gateway_url": gateway_url,
        "MerchantID":  merchant_id,
        "TradeInfo":   trade_info,
        "TradeSha":    trade_sha,
        "Version":     "2.0",
    }


def parse_notify(form_data: dict, hash_key: str, hash_iv: str) -> dict | None:
    """
    解析藍新 Webhook 通知，驗證 TradeSha 後回傳解密結果。
    form_data 為 request.form() 的結果。
    """
    status     = form_data.get("Status", "")
    trade_info = form_data.get("TradeInfo", "")
    trade_sha  = form_data.get("TradeSha", "")

    # 驗證 TradeSha
    expected_sha = sha256_mac(trade_info, hash_key, hash_iv)
    if expected_sha != trade_sha:
        logging.warning(f"[NewebPay] TradeSha mismatch: expected={expected_sha} got={trade_sha}")
        return None

    if status != "SUCCESS":
        logging.info(f"[NewebPay] Payment not success: status={status}")
        return None

    # 解密 TradeInfo
    try:
        decrypted = aes_decrypt(trade_info, hash_key, hash_iv)
        result    = dict(urllib.parse.parse_qsl(decrypted))
        return result
    except Exception as e:
        logging.error(f"[NewebPay] Decrypt failed: {e}")
        return None
