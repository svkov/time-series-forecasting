import os

import click


def generate_header(body):
    if not isinstance(body, str):
        raise ValueError(f'body должен быть строкой, вместо этого {type(body)}')
    return f"""
    \\documentclass[12pt]{{article}}
    \\usepackage{{graphicx}}
    \\begin{{document}}
    {body}
    \\end{{document}}
    """


def generate_figure(path, name, label):
    return f"""
    
    \\begin{{figure}}[ht]
    \\begin{{center}}
    \\scalebox{{0.4}}{{
       \\includegraphics{{{path}}}
    }}

    \\caption{{
    \\label{{{label}}}
         {name}.}}
    \\end {{center}}
    \\end {{figure}}
    
    """


def concat_parts(*parts):
    for part in parts:
        if not isinstance(part, str):
            raise ValueError(f'Тип куска: {type(part)}, должно быть str')
    return '\n'.join(parts)


@click.command()
@click.option('--input')
@click.option('--output')
def generate(input, output):
    path_to_figures = input.replace('\\', '/').split()
    name = 'Какое-то название'
    figs = [generate_figure(f'../{path}', name, f'graph{i}') for i, path in enumerate(path_to_figures)]
    body = concat_parts(*figs)
    # tex = generate_header(body)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(body)


if __name__ == '__main__':
    generate()  # noqa
