import _sitebuiltins
import argparse
import asyncio
from dotenv import load_dotenv
from pathlib import Path
import playwright
from playwright.async_api import async_playwright
import re
import os
import time

# _XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX 패턴
UUID_PATTERN = re.compile(
    r'_[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}',
    re.IGNORECASE
)

# 파일명 전체가 UUID 형식인 경우: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
UUID_FILENAME_PATTERN = re.compile(
    r'^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}',
    re.IGNORECASE
)

async def download_all_photos(
    url: str,
    site: int,
    email: str,
    password: str,
    skip: int = 0,
    output_dir: str = "./downloads",
    headless: bool = False,
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # 로그인 필요시 False로 설정
            downloads_path="./downloads"
        )
        
        context = await browser.new_context(
            accept_downloads=True
        )
        page = await context.new_page()

        # ── 로그인 ─────────────────────────────────────────────
        print("🔐 로그인 중...")
        #login_url = "https://solvation.app.sitemark.com/login"
        login_url = f"https://solvation.app.sitemark.com/login?redirect=%2Foperations%2F{site}%2Fphoto"
        await page.goto(login_url, wait_until="networkidle")
        await page.fill("input[type='email']", email)
        await page.fill("input[type='password']", password)

        #await page.click("button[type='submit']")
        # 방법 1: CSS 클래스로 클릭 (권장) <- 정상적으로 로그인 안됨
        #await page.click('button.sm-btn-intent-primary')

        # 방법 2: type=submit으로 클릭 <- 정상적으로 로그인 안됨
        #await page.wait_for_timeout(300)  # 입력 완료 대기
        #await page.click('button[type="submit"]')

        # 방법 3: 클릭 후 네트워크 응답 대기 <- 로그인됨
        async with page.expect_response(lambda r: '/login' in r.url or '/auth' in r.url) as resp:
            await page.click('button[type="submit"]')
        response = await resp.value
        print(response.status)  # 200이면 성공

        await page.wait_for_load_state("networkidle")
        print("✅ 로그인 완료")
        
        # 페이지 접속
        await page.goto(url)
        
        # 로그인이 필요한 경우 수동으로 처리할 시간 부여
        print("로그인이 필요하면 30초 내에 완료해 주세요...")
        await page.wait_for_timeout(10000)
        
        # 첫 번째 사진 태그에서 전체 사진 수 파악
        photo_tag = await page.wait_for_selector(".sm-tag", timeout=10000)
        photo_text = await photo_tag.inner_text()
        # "사진 1/1818" 형식에서 전체 수 추출
        total_photos = int(photo_text.split("/")[1].strip())
        print(f"전체 사진 수: {total_photos}")
        
        downloaded_count = 0
        
        i = 0
        while True:
            try:
                print(f"[{i+1}/{total_photos}] 다운로드 시작...")
                
                if i >= skip:
                    # 현재 사진 번호 확인
                    photo_tag = await page.wait_for_selector(".sm-tag", timeout=5000)
                    photo_text = await photo_tag.inner_text()
                    print(f"  현재 페이지: {photo_text}")

                    # 다운로드 버튼 클릭 및 다운로드 대기
                    async with page.expect_download(timeout=30000) as download_info:
                        download_btn = await page.wait_for_selector(
                            "button.download-photo-btn", 
                            timeout=5000
                        )
                        await download_btn.click()
                    
                    download = await download_info.value
                    
                    # 파일 저장 (원본 파일명 사용)
                    suggested_filename = UUID_PATTERN.sub('', download.suggested_filename).upper()
                    if not suggested_filename:
                        suggested_filename = f"photo_{i+1:04d}.jpg"
                    
                    #save_path = f"./downloads/{i+1:04d}_{suggested_filename}"
                    save_path = f"{output_dir}/{site}/TM/{suggested_filename}" if "_T.JPG" in suggested_filename \
                                else f"{output_dir}/{site}/RGB/{suggested_filename}"
                    await download.save_as(save_path)
                    downloaded_count += 1
                    print(f"  저장 완료: {save_path}")

                    #files = _list_files(target_dir = output_dir)
                    count = await _delete_files(target_dir = output_dir)

                    #time.sleep(1.0)
                
                # 다음 사진으로 이동 (chevron-right 클릭)
                chevron_right = await page.wait_for_selector(
                    "div.chevron-align-middle.pull-right",
                    timeout=5000
                )
                
                await chevron_right.click()
                
                # 이미지 로딩 대기
                await page.wait_for_timeout(1500)
                
                i += 1
            
            except playwright._impl._errors.TimeoutError as e:
                print(f"  오류 발생 ({i+1}번째): {e}")
                if 'div.chevron-align-middle.pull-right' in str(e):
                    break
                
            except Exception as e:
                print(f"  오류 발생 ({i+1}번째): {e}")
                # 오류 시 잠시 대기 후 계속 진행
                await page.wait_for_timeout(3000)
                continue
            #finally:
            #    continue
        
        print(f"\n완료! 총 {downloaded_count}/{total_photos}장 다운로드")
        await browser.close()

async def _list_files(target_dir: str) -> list[Path]:
    """UUID 형식 파일 목록 반환"""
    base = Path(target_dir)
    if not base.is_dir():
        print(f"오류: '{target_dir}' 폴더를 찾을 수 없습니다.")
        return []

    files = sorted([f for f in base.iterdir() if f.is_file() and await _is_uuid_file(f)])

    if not files:
        print("  UUID 형식 파일 없음")
    for i, f in enumerate(files, 1):
        size = f.stat().st_size
        #print(f"  {i:3}. {f.name:<55} {size:>10,} bytes")

    return files

async def _is_uuid_file(file: Path) -> bool:
    """파일명이 UUID 형식으로 시작하는지 확인"""
    return bool(UUID_FILENAME_PATTERN.match(file.name))

async def _delete_files(target_dir: str, dry_run: bool = False) -> int:
    """UUID 형식 파일 일괄 삭제"""
    files = await _list_files(target_dir)
    if not files:
        return 0

    count = 0
    for f in files:
        if dry_run:
            print(f"  [DRY RUN] 삭제 예정: {f.name}")
        else:
            f.unlink()
            print(f"  삭제 완료: {f.name}")
        count += 1

    #print(f"\n{'삭제 예정' if dry_run else '삭제 완료'}: 총 {count}개")
    return count

async def main():
    load_dotenv()

    email = os.getenv("SITEMARK_EMAIL")
    password = os.getenv("SITEMARK_PASSWORD")

    ap = argparse.ArgumentParser(description="Sitemark에서 사진 다운로드")
    ap.add_argument("--site-name",   default="EWP-서오창IC-2")
    ap.add_argument("--site",        type=int, default=29696)
    ap.add_argument("--skip",        type=int, default=0)
    ap.add_argument("--output-dir",  default="./downloads")

    args = ap.parse_args()

    sites = [
        ("EWP-서오창IC-2", 27662, "서오창IC"),
        ("갈평저수지", 27846, "갈평저수지"),
        ("옥산 1호", 26146, "청주휴게소"),
        ("에스엘에너지_사천시", 27622, "에스엘에너지_사천시"),
        ("Site-1", 29695, "송원대학교 운동장"),
        ("Site-2", 29696, "송원대학교 강의동A"),
        ("Site-2", 29719, "송원대학교 강의동A"),
        ("환경관리_300KW", 30022, "장흥군그린환경센터"),
        ("K_Demo", 32645, "타이어테크 기아자동차화성공장점"),
        ("Jeju", 20568, "제주 서귀포 토산리")
    ]

    #url = "https://solvation.app.sitemark.com/operations/27662/photo" # EWP-서오창IC-2, 1818개 (서오창IC), 좌표: 36.727747107, 127.416422423
    # https://map.naver.com/p/search/충북%20청주시%20청원구%20오창읍%20성산리%20산145-33/address/3zA2C0,2AcQvh,충청북도%20청주시%20청원구%20오창읍%20성산리%20산145-33?c=15.00,0,0,2,dh&isCorrectAnswer=true
    
    #url = "https://solvation.app.sitemark.com/operations/27846/photo" # 갈평저수지, 488개 (갈평저수지), 좌표: 36.452427985, 127.838110788
    # https://map.naver.com/p/search/갈평저수지/place/17816850?c=18.53,0,0,2,dh&placePath=/home?from=map&fromPanelNum=2&timestamp=202605260839&locale=ko&svcName=map_pcv5&searchText=갈평저수지
    
    #url = "https://solvation.app.sitemark.com/operations/26146/photo" # 옥산 1호, 2050개 (청주휴게소), 좌표: 36.717732965, 127.345088011
    # https://map.naver.com/p/search/충북%20청주시%20흥덕구%20옥산면%20사정리%20398-5/address/3zx57E,2AcqQb,충청북도%20청주시%20흥덕구%20옥산면%20사정리%20398-5?c=16.00,0,0,2,dh&isCorrectAnswer=true
    
    #url = "https://solvation.app.sitemark.com/operations/27622/photo" # 에스엘에너지_사천시, 632개 (경남 사천시 서포면 구평리 92-70), 좌표: 34.994684025, 127.994609347
    # https://map.naver.com/p/search/경남%20사천시%20서포면%20구평리%2092-70/address/3zYjN2,2z28p6,경상남도%20사천시%20서포면%20구평리%2092-70?c=18.32,0,0,2,dh&isCorrectAnswer=true
    
    #url = "https://solvation.app.sitemark.com/operations/29695/photo" # Site-1, 372개 (송원대학교 운동장), 좌표: 35.107644582, 126.873318218
    # https://map.naver.com/p/search/송원대학교교운동장/place/16293908?c=15.00,0,0,2,dh&isCorrectAnswer=true&placePath=/home?from=map&fromPanelNum=1&additionalHeight=76&timestamp=202605261125&locale=ko&svcName=map_pcv5&searchText=송원대학교교운동장
    
    #url = "https://solvation.app.sitemark.com/operations/29696/photo" # Site-2, 414개 (송원대학교강의동A), 좌표: 35.108713926, 126.874793831
    # https://map.naver.com/p/search/송원대학교강의동A/place/16294393?c=15.00,0,0,2,dh&isCorrectAnswer=true&placePath=/home?from=map&fromPanelNum=1&additionalHeight=76&timestamp=202605261126&locale=ko&svcName=map_pcv5&searchText=송원대학교강의동A
    #url = "https://solvation.app.sitemark.com/operations/29719/photo" # Site-2, 338개 (송원대학교강의동A), 좌표: 35.108713926, 126.874793831

    #url = "https://solvation.app.sitemark.com/operations/30022/photo" # 환경관리_300KW, 760개 (장흥군그린환경센터), 좌표: 34.710885527, 126.922004634
    # https://map.naver.com/p/search/전남%20장흥군%20부산면%20부춘리%2091/address/3zfjDI,2yQe0h,전라남도%20장흥군%20부산면%20부춘리%2091?c=16.00,0,0,2,dh&isCorrectAnswer=true
    
    #url = "https://solvation.app.sitemark.com/operations/32645/photo" # K_Demo, 1012개 (타이어테크 기아자동차화성공장점), 좌표: 37.032311154, 126.786649997
    # https://map.naver.com/p/search/경기%20화성시%20만세구%20우정읍%20이화리%201714-1/address/3z9GCr,2ApMCv,경기도%20화성시%20만세구%20우정읍%20이화리%201714-1?c=16.05,0,0,2,dh&isCorrectAnswer=true

    site = args.site
    site_name = args.site_name
    if site_name != "EWP-서오창IC-2":
        # site_name이 변경된 경우: 부분 문자열 매칭(대소문자 무시)으로 site 찾기
        matches = [s for s in sites if site_name.lower() in s[2].lower() or site_name.lower() in s[0].lower()]
        if not matches:
            raise ValueError(f"'{site_name}'과 일치하는 사이트를 찾을 수 없습니다.")
        if len(matches) > 1:
            matched_names = ", ".join(f"{s[2]}({s[1]})" for s in matches)
            raise ValueError(f"'{site_name}'과 일치하는 사이트가 여러 개입니다: {matched_names}")

        if len(matches) > 1:
            matched_names = ", ".join(f"{s[2]}({s[1]})" for s in matches)
            raise ValueError(f"'{site_name}'과 일치하는 사이트가 여러 개입니다: {matched_names}")
            
        site = int(matches[0][1])
        site_name = matches[0][2]
    elif site != 29696:
        # site가 변경된 경우: 정확한 숫자 매칭으로 site_name 찾기
        matches = [s for s in sites if int(s[1]) == site]
        if not matches:
            raise ValueError(f"site={site}와 일치하는 사이트를 찾을 수 없습니다.")
        site_name = matches[0][2]

    url = f"https://solvation.app.sitemark.com/operations/{site}/photo"
    await download_all_photos(
        url=url,
        site=site,
        email=email,
        password=password,
        skip = args.skip,
        output_dir=args.output_dir,
        headless=True,
    )


if __name__ == "__main__":
    asyncio.run(main())