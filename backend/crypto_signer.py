from __future__ import annotations

import re
import secrets
from pathlib import Path

from gmssl import func, sm2 as gm_sm2, sm3

# 随机数合规说明（整改记录 2026-09-30）：
# 原实现使用 gmssl.func.random_hex，其底层为 Python `random.choice`
# （Mersenne Twister，非密码学安全 PRNG），用于私钥生成与 SM2 签名随机数 k。
# 按 NIST SP 800-90A、FIPS 186-5 与 GM/T 0003，椭圆曲线签名的随机数必须取自
# 经认可的密码学安全随机源（CSPRNG），否则签名 k 可被重建并导致私钥恢复。
# 此处改用 `secrets`（直接向操作系统 CSPRNG 取熵），并将 k 提升为 256 位，
# 与 SM2 推荐曲线的群阶 [1, n-1] 取值范围一致。
_SECRET_NONCE_HEXLEN = 64  # 256 bit，SM2 推荐曲线群阶位长
_SECRET_KEY_HEXLEN = 64    # 256 bit


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _is_hex(value, length: int) -> bool:
    """十六进制格式校验：长度须精确匹配且仅含十六进制字符。"""
    return (isinstance(value, str) and len(value) == length
            and bool(_HEX_RE.match(value)))


def _secure_hex(n_bytes: int) -> str:
    """从操作系统 CSPRNG 生成 n_bytes 字节的十六进制随机串。"""
    return secrets.token_hex(n_bytes)


def load_or_create_signer(key_path: str | Path) -> "SM2Signer":
    path = Path(key_path)
    if path.is_file():
        private_key = path.read_text(encoding="utf-8").strip()
        if len(private_key) == 64:
            return SM2Signer(private_key)
        import logging

        logging.getLogger(__name__).warning(
            "演示 SM2 私钥长度非法（%d 位，应为 64），重新生成", len(private_key)
        )
    private_key = _secure_hex(_SECRET_KEY_HEXLEN // 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(private_key, encoding="utf-8")
    return SM2Signer(private_key)


class SM2Signer:
    """国密 SM2 签名器（gmssl 纯 Python）。

    对应论文第 5 章 P3 命题：签名保证单张凭证真实性（不可伪造、不可抵赖），
    哈希链保证整条序列连续性。二者缺一不可。
    """

    def __init__(self, private_key_hex: str) -> None:
        if len(private_key_hex) != 64:
            raise ValueError(f"SM2 私钥须为 64 位十六进制，收到 {len(private_key_hex)} 位")
        self.private_key = private_key_hex
        self.public_key = self._derive_public_key()
        self._sm2 = gm_sm2.CryptSM2(
            private_key=self.private_key, public_key=self.public_key
        )

    def _derive_public_key(self) -> str:
        tmp = gm_sm2.CryptSM2(private_key=self.private_key, public_key="")
        return tmp._kg(int(self.private_key, 16), tmp.ecc_table["g"])

    def sign(self, message: str) -> str:
        # 签名随机数 k 取自操作系统 CSPRNG，256 位，符合 SM2 群阶取值范围
        return self._sm2.sign(message.encode("utf-8"), _secure_hex(_SECRET_NONCE_HEXLEN // 2))

    def verify(self, message: str, signature: str) -> bool:
        return self._sm2.verify(signature, message.encode("utf-8"))

    def fingerprint(self) -> str:
        return sm3.sm3_hash(func.bytes_to_list(self.public_key.encode()))[:16]

    def public_key_hex(self) -> str:
        return self.public_key


# --------------------------------------------------------------------------------------
# G4 改造（2026-09-30）：节点级密钥隔离与验签/签名职责分离
#
# 原实现的问题
#   engine.DEFAULT_SM2_KEY 为全局唯一密钥路径，主链与 12 条节点链全部使用同一对密钥，
#   导致任一节点（或持有该密钥文件的一方）可伪造其余 11 个节点的凭证，节点级来源
#   真实性与不可否认性不成立；且 verify_chain 依赖持私钥的 signer 对象，验签方必须
#   同时持有私钥，与"公开验签"的机制描述不符。
#
# 本次改造
#   1) 每节点一把独立密钥（keys/node_<node_id>.hex），主链按凭证所属节点选择签名者；
#   2) 新增 SM2Verifier——仅凭公钥即可验签，验签方不再需要私钥；
#   3) 新增 load_node_signers / export_public_keyring，提供节点公钥指纹清单，
#      作为后续链尾外部锚定（G3：Merkle 根 + 哈希公示）的信任锚材料。
# --------------------------------------------------------------------------------------


class SM2Verifier:
    """仅凭公钥完成验签的验证器（不持有私钥）。

    对应审计发现 G4-a：原实现中 verify_chain 复用签名器对象，验签方被迫持有私钥。
    """

    def __init__(self, public_key_hex: str) -> None:
        if len(public_key_hex) != 128:
            raise ValueError(f"SM2 公钥须为 128 位十六进制，收到 {len(public_key_hex)} 位")
        self.public_key = public_key_hex
        self._sm2 = gm_sm2.CryptSM2(private_key="", public_key=public_key_hex)
        # 可观测计数：区分「验签判定不通过」与「验签过程本身抛异常」。
        # 两者返回值都是 False（fail-closed），但含义完全不同——前者是攻击者
        # 提供了无效签名，后者意味着底层实现对一条**合法**凭证失效。
        # 实测中后者为偶发（本审计在 30,000 次裸签名-验签中未复现，
        # 但在多次实验运行中观察到基线链被误判），故须可量化而非静默。
        self.stats = {"ok": 0, "invalid": 0, "exception": 0}

    def verify(self, message: str, signature: str) -> bool:
        """验签，失败一律返回 False（fail-closed）。

        健壮性说明：gmssl 的 CryptSM2.verify 对无效签名**不返回 False，而是抛异常**
        （典型如 _double_point 中 Point 为 None 时报 TypeError）。而本方法处理的是
        来自不可信输入的凭证——若把异常直接上抛，攻击者提交一条签名畸形的凭证即可使
        核验服务抛出未捕获异常，构成拒绝服务。因此这里做两道防护：

          1) 前置格式校验：签名与公钥须为 128 位十六进制，长度或字符不符直接判失败；
          2) 捕获底层库抛出的异常并同样判为失败。

        安全语义不变——"无法确认"与"确认无效"在本方法中都记为未通过。
        """
        if not _is_hex(signature, 128) or not _is_hex(self.public_key, 128):
            self.stats["invalid"] += 1
            return False
        try:
            ok = bool(self._sm2.verify(signature, message.encode("utf-8")))
        except Exception:
            # 覆盖 gmssl 的 TypeError / ValueError / IndexError 等
            self.stats["exception"] += 1
            return False
        self.stats["ok" if ok else "invalid"] += 1
        return ok

    def fingerprint(self) -> str:
        """公钥指纹（SM3 前 128 位十六进制），用于公钥清单与后续外部锚定。"""
        return sm3.sm3_hash(func.bytes_to_list(self.public_key.encode()))[:32]

    @staticmethod
    def fingerprint_of(text: str) -> str:
        """任意文本的 SM3 指纹（前 128 位十六进制），用于清单聚合指纹。"""
        return sm3.sm3_hash(func.bytes_to_list(text.encode("utf-8")))[:32]


def aggregate_keyring_fingerprint(pubkeys: dict) -> str:
    """公钥清单的聚合指纹（node_id -> 公钥十六进制）。

    对 node_id 排序后拼接 "node_id:公钥" 再取 SM3，使结果与清单的书写顺序、
    字典迭代顺序无关。engine.keyring_fingerprint 与 chain.verify_chain 的清单校验
    都委托本函数，避免同一口径出现两份实现而各自漂移。

    参数为 {node_id: public_key_hex} 映射；engine 侧的 keyring 结构
    （node_id -> {"public_key": ..., "fingerprint": ...}）取其 public_key 后传入。
    """
    parts = [f"{nid}:{pk}" for nid, pk in sorted(pubkeys.items())]
    return aggregate_fingerprint("|".join(parts))


def aggregate_fingerprint(text: str) -> str:
    """SM3 指纹（模块级别名，便于在不持有验证器对象时调用）。"""
    return SM2Verifier.fingerprint_of(text)


def _node_key_path(keys_dir: str | Path, node_id: str) -> Path:
    safe = "".join(ch for ch in str(node_id) if ch.isalnum() or ch in "-_.")
    return Path(keys_dir) / f"node_{safe}.hex"


def load_or_create_node_signer(keys_dir: str | Path, node_id: str) -> SM2Signer:
    """载入（或首次生成）某节点专属的 SM2 签名器。"""
    path = _node_key_path(keys_dir, node_id)
    return load_or_create_signer(path)


def load_node_signers(keys_dir: str | Path, node_ids) -> dict[str, SM2Signer]:
    """为每个节点建立独立签名器。"""
    return {str(nid): load_or_create_node_signer(keys_dir, nid) for nid in node_ids}


def export_public_keyring(signers: dict) -> dict:
    """导出节点公钥清单：node_id -> {public_key, fingerprint}。

    该清单是 G3 外部锚定的信任锚材料——清单本身被哈希公示后，
    攻击者既无法替换公钥，也无法伪造他节点凭证而不被发现。
    """
    ring = {}
    for nid, sg in signers.items():
        pk = sg.public_key_hex() if isinstance(sg, SM2Signer) else sg.public_key
        ring[str(nid)] = {
            "public_key": pk,
            "fingerprint": SM2Verifier(pk).fingerprint(),
        }
    return ring


def verifiers_from_keyring(keyring: dict) -> dict[str, SM2Verifier]:
    return {nid: SM2Verifier(v["public_key"]) for nid, v in keyring.items()}
