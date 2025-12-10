"""
AES加密测试脚本
用于验证密码加密功能是否正常工作
"""
import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def aes_encrypt_password(password: str) -> str:
    """
    使用AES CBC模式加密密码

    Args:
        password: 明文密码

    Returns:
        str: Base64编码的加密密文
    """
    # AES密钥和IV（固定值）
    AES_KEY = b'JzjPLY9632AijnEQ'  # 16字节
    AES_IV = b'DYgjCEIikmj2W9xN'   # 16字节

    # 创建AES加密器（CBC模式）
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)

    # PKCS7填充
    padded_data = pad(password.encode('utf-8'), AES.block_size)

    # 加密
    encrypted_data = cipher.encrypt(padded_data)

    # Base64编码
    encrypted_base64 = base64.b64encode(encrypted_data).decode('utf-8')

    return encrypted_base64


def aes_decrypt_password(encrypted_base64: str) -> str:
    """
    解密密码（用于测试验证）

    Args:
        encrypted_base64: Base64编码的密文

    Returns:
        str: 明文密码
    """
    # AES密钥和IV（固定值）
    AES_KEY = b'JzjPLY9632AijnEQ'  # 16字节
    AES_IV = b'DYgjCEIikmj2W9xN'   # 16字节

    # Base64解码
    encrypted_data = base64.b64decode(encrypted_base64)

    # 创建AES解密器（CBC模式）
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)

    # 解密
    decrypted_padded = cipher.decrypt(encrypted_data)

    # 去除填充
    decrypted_data = unpad(decrypted_padded, AES.block_size)

    return decrypted_data.decode('utf-8')


def test_encryption():
    """测试加密和解密"""
    print("=" * 60)
    print("AES加密测试")
    print("=" * 60)
    print()

    # 测试用例
    test_passwords = [
        "admin123",
        "password",
        "Test@123",
        "very_long_password_1234567890",
        "短密码",  # 中文密码
    ]

    print(f"加密配置:")
    print(f"  算法: AES")
    print(f"  模式: CBC")
    print(f"  填充: PKCS7")
    print(f"  密钥长度: 128位")
    print(f"  密钥: JzjPLY9632AijnEQ")
    print(f"  IV: DYgjCEIikmj2W9xN")
    print()

    for i, password in enumerate(test_passwords, 1):
        print(f"测试 {i}: {password}")
        print("-" * 60)

        # 加密
        encrypted = aes_encrypt_password(password)
        print(f"  明文: {password}")
        print(f"  密文(Base64): {encrypted}")

        # 解密验证
        try:
            decrypted = aes_decrypt_password(encrypted)
            if decrypted == password:
                print(f"  ✓ 验证成功: 解密结果与原文匹配")
            else:
                print(f"  ✗ 验证失败: 解密结果不匹配")
                print(f"    期望: {password}")
                print(f"    实际: {decrypted}")
        except Exception as e:
            print(f"  ✗ 解密失败: {e}")

        print()

    print("=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    test_encryption()
