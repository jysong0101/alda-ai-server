import os
import logging
from datetime import datetime

LOGS_DIRECTORY = "logs"

deleted_files_count = 0

# 디렉토리 내 파일 확인
for filename in os.listdir(LOGS_DIRECTORY):
    file_path = os.path.join(LOGS_DIRECTORY, filename)

    # 파일 삭제
    os.remove(file_path)
    deleted_files_count += 1