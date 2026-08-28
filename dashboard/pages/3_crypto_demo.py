"""
LIVE: Crypto Pipeline Demo

Demonstrates the EveFL cryptographic stack.

- AES-256-GCM: REAL implementation (ClassicalCipherSuite)
- CKKS: STUB / DEMO ONLY
- Groth16: STUB / DEMO ONLY
"""

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evefl.crypto.classical import ClassicalCipherSuite
from evefl.crypto.base import EncryptedPayload

st.set_page_config(page_title="Crypto Demo", page_icon="🔒")

st.title("🔒 Crypto Pipeline Demo")

st.markdown("""
EveFL protects model updates using a **Classical Cipher Suite**:

```
Raw key material (BB84 sifted key)
    ↓
HKDF-SHA256 key derivation
    ↓
AES-256-GCM encryption
```

This page demonstrates a live encrypt/decrypt round-trip.
""")

# ---------------------------------------------------------------------------
# AES-256-GCM (REAL)
# ---------------------------------------------------------------------------

st.divider()
st.header("1. AES-256-GCM — Real Implementation ✅")

suite = ClassicalCipherSuite()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Encrypt")
    plaintext = st.text_area("Plaintext message", "Hello from EveFL client 0", height=100)
    key_material = st.text_input("Key material (hex)", "abcd1234" * 8, help="Simulated BB84 sifted key as hex string")

    if st.button("🔐 Encrypt", key="enc"):
        try:
            raw_key = bytes.fromhex(key_material)
        except ValueError:
            st.error("Invalid hex string. Use only 0-9 and a-f.")
            st.stop()

        derived_key = suite.derive_key(raw_key, info=b"evefl-demo")
        payload = suite.encrypt(plaintext.encode("utf-8"), derived_key)

        st.session_state["payload"] = payload
        st.session_state["derived_key"] = derived_key
        st.session_state["plaintext"] = plaintext

        st.success("Encrypted!")
        st.json({
            "ciphertext (hex)": payload.ciphertext.hex()[:64] + "...",
            "nonce (hex)": payload.nonce.hex(),
            "tag": "embedded in ciphertext (AES-GCM)" if payload.tag is None else payload.tag.hex(),
            "derived_key (hex)": derived_key.hex(),
        })

with col2:
    st.subheader("Decrypt")
    if "payload" in st.session_state:
        if st.button("🔓 Decrypt", key="dec"):
            decrypted = suite.decrypt(st.session_state["payload"], st.session_state["derived_key"])
            st.success(f"Decrypted: **{decrypted.decode('utf-8')}**")
            if decrypted.decode("utf-8") == st.session_state["plaintext"]:
                st.balloons()
    else:
        st.info("Encrypt something first!")

# ---------------------------------------------------------------------------
# CKKS (STUB)
# ---------------------------------------------------------------------------

st.divider()
st.header("2. CKKS Homomorphic Encryption — STUB / DEMO ONLY ⚠️")

st.error("""
**This is a research placeholder. CKKS is NOT yet implemented in EveFL.**

The repository contains `tenseal==0.3.17` in requirements, but no CKKS
integration exists for the FL pipeline. This section is for demonstration
purposes only.
""")

st.markdown("""
**What CKKS would do:**
- Encrypt model gradients so the server never sees plaintext updates
- Allow computation on encrypted vectors (homomorphic addition for FedAvg)
- Current status: **Not integrated**
""")

st.button("Run CKKS Demo", disabled=True, help="Stub — not implemented")

# ---------------------------------------------------------------------------
# Groth16 (STUB)
# ---------------------------------------------------------------------------

st.divider()
st.header("3. Groth16 Zero-Knowledge Proofs — STUB / DEMO ONLY ⚠️")

st.error("""
**This is a research placeholder. Groth16 is NOT yet implemented in EveFL.**

No ZK-proof system is currently wired into the aggregation pipeline.
This section is for demonstration purposes only.
""")

st.markdown("""
**What Groth16 would do:**
- Prove that a client's update was computed correctly without revealing weights
- Verify proof at the server before aggregation
- Current status: **Not integrated**
""")

st.button("Run Groth16 Demo", disabled=True, help="Stub — not implemented")

st.divider()
st.caption("""
For the actual FL experiment, only **AES-256-GCM** is active.
CKKS and Groth16 are stretch goals for future work.
""")