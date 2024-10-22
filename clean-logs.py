import os
import logging
from datetime import datetime

LOGS_DIRECTORY = "logs"

deleted_files_count = 0

log_filename = f"/logs-clean/clean_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting log cleanup process...")

# 디렉토리 내 파일 확인
for filename in os.listdir(LOGS_DIRECTORY):
    file_path = os.path.join(LOGS_DIRECTORY, filename)

    try:
        # 파일 삭제
        os.remove(file_path)
        logging.info(f"Deleted log file: {filename}")
        deleted_files_count += 1
    except Exception as e:
        logging.error(f"Error while processing file {filename}: {e}")

logging.info(f"Log cleanup completed. Total files deleted: {deleted_files_count}")
