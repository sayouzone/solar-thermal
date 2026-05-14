"""
file_hasher.py - 파일 내용을 다양한 해시 알고리즘으로 변환하는 유틸리티
"""

import hashlib
import argparse
from pathlib import Path


ALGORITHMS = ["md5", "sha1", "sha256", "sha512", "sha3_256"]


def hash_file(file_path: str | Path, algorithm: str = "sha256") -> str:
    """
    파일 내용을 읽어 해시값을 반환합니다.
    대용량 파일도 청크 단위로 처리합니다.
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"지원하지 않는 알고리즘: {algorithm}")

    hasher = hashlib.new(algorithm)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    with path.open("rb") as f:
        while chunk := f.read(8192):  # 8KB 청크 단위
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_file_all(file_path: str | Path) -> dict[str, str]:
    """파일에 대해 모든 주요 알고리즘의 해시값을 반환합니다."""
    return {algo: hash_file(file_path, algo) for algo in ALGORITHMS}


def hash_content(content: str | bytes, algorithm: str = "sha256") -> str:
    """문자열 또는 바이트 내용을 해시값으로 변환합니다."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.new(algorithm, content).hexdigest()


def verify_file(file_path: str | Path, expected_hash: str, algorithm: str = "sha256") -> bool:
    """파일의 해시값이 기대값과 일치하는지 검증합니다."""
    actual = hash_file(file_path, algorithm)
    return actual.lower() == expected_hash.lower()


def main():
    parser = argparse.ArgumentParser(description="파일 해시 계산기")
    #parser.add_argument("file", help="해시를 계산할 파일 경로")
    parser.add_argument("path", help="해시를 계산할 파일 경로")
    parser.add_argument(
        "-a", "--algorithm",
        choices=ALGORITHMS,
        default="sha256",
        help="해시 알고리즘 선택 (기본값: sha256)",
    )
    parser.add_argument("--all", action="store_true", help="모든 알고리즘으로 해시 계산")
    parser.add_argument("--verify", metavar="HASH", help="파일 해시값 검증")

    args = parser.parse_args()

    image_path = Path(args.path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    
    # 지원 확장자 목록
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    # 이미지 파일 목록 수집
    if image_path.is_dir():
        image_files = sorted([
            p for p in image_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])
    else:
        # 단일 파일인 경우
        image_files = [image_path]

    try:
        if args.verify:
            for img in image_files:
                full_path = image_path / img.name
                ok = verify_file(full_path, args.verify, args.algorithm)
                status = "✅ 일치" if ok else "❌ 불일치"
                print(f"검증 결과: {status}")
                print(f"  기대값: {args.verify}")
                print(f"  실제값: {hash_file(args.file, args.algorithm)}")

        elif args.all:
            print(f"파일: {args.file}\n")
            for algo, digest in hash_file_all(args.file).items():
                print(f"  {algo:<12}: {digest}")

        else:
            for img in image_files:
                full_path = image_path / img.name
                digest = hash_file(full_path, args.algorithm)
                print(f"{args.algorithm}({full_path}) = {digest}")

    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
