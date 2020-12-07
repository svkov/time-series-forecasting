from collections import Iterable


def generate_table_header(body, caption, columns_width, label='result_table'):
    if not isinstance(body, str):
        raise ValueError(f'body должен быть строкой, вместо этого {type(body)}')
    width = ''.join([f'|p{{{width}cm}}' for width in columns_width]) + '|'
    return \
        f"""\\begin{{center}}
    \\begin{{longtable}}{{{width}}}
    \\caption{{{caption}}}\\label{{{label}}}\\\\
    {body}
\\end{{longtable}}
\\end{{center}}"""


def generate_row(name, row, sep='\\hline\n', end='\\\\\n'):
    if isinstance(name, list) or isinstance(name, tuple):
        name = list(name)
        name = f'{name[0]} & {name[1]} & '
    else:
        name = f'{name} & '

    def process_row_value(x):
        if isinstance(x, float):
            return str(round(x, 1))
        return str(x)

    if not isinstance(row, list):
        row = row.tolist()
    values = ' & '.join(map(process_row_value, row))
    return sep + name + values + end


def generate_columns(index_names, columns):
    columns = list(index_names) + list(columns)
    return '\\hline\n' + ' & '.join([f'\\textbf{{{name}}}' for name in columns]) + '\\\\\n'


def save_latex(content, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
