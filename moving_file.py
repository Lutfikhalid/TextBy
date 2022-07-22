import os

target_folder = r'C:\Users\Lutfi Khalid\PycharmProjects\pythonProject\TextBy\UserForms' + '\\'
source_folder = r'C:\Users\Lutfi Khalid\PycharmProjects\pythonProject\TextBy\venv' + '\\'

for path, dir, files in os.walk(source_folder):
    print(path)
    print(files)