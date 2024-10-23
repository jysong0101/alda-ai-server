import os
import logging

LOGS_DIRECTORY = "server-logs"
TEST_LOGS_DIRECTORY = "test-logs"

for filename in os.listdir(LOGS_DIRECTORY):
    file_path = os.path.join(LOGS_DIRECTORY, filename)

    os.remove(file_path)
    
for filename in os.listdir(TEST_LOGS_DIRECTORY):
    file_path = os.path.join(TEST_LOGS_DIRECTORY, filename)

    os.remove(file_path)