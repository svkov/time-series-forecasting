import os
import subprocess


def generate():
    with open(os.devnull, 'wb') as devnull:
        subprocess.check_call(['pdflatex', 'main_example.tex'], stdout=devnull, stderr=subprocess.STDOUT)


def main():
    """
    Нужно компилировать документ два раза, чтобы появилось содержание и ссылки.
    На всякий случай будем компилировать три раза.
    """
    os.chdir('spbu_diploma/')
    for i in range(3):
        generate()


if __name__ == '__main__':
    main()
