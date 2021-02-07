import os
import subprocess


def generate():
    with open(os.devnull, 'wb') as devnull:
        # subprocess.check_call(['pdflatex', 'main_example.tex'], stdout=devnull, stderr=subprocess.STDOUT)
        subprocess.check_call(['pdflatex', 'main_example.tex'])


def main():
    """
    Нужно компилировать документ два раза, чтобы появилось содержание и ссылки.
    На всякий случай будем компилировать три раза.
    """
    os.chdir('spbu_diploma/')
    n = 3
    for i in range(n):
        generate()
        print(f'compile [{i + 1}/{n}] times')


if __name__ == '__main__':
    main()
