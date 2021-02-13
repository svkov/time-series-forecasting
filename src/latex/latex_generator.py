import pandas as pd


class LatexGenerator:
    content = ''

    def __init__(self, content, **kwargs):
        self.content = content

    def generate_header(self, body):
        if not isinstance(body, str):
            raise ValueError(f'body должен быть строкой, вместо этого {type(body)}')
        self.content += f"""
                        \\documentclass[12pt]{{article}}
                        \\usepackage{{graphicx}}
                        \\begin{{document}}
                            {body}
                        \\end{{document}}
                        """

    def concat_parts(self, *parts):
        for part in parts:
            if not isinstance(part, str):
                raise ValueError(f'Тип куска: {type(part)}, должно быть str')
        self.content += '\n'.join(parts)

    def save_latex(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.content)


class LatexPictureGenerator(LatexGenerator):

    def __init__(self, content, path, **kwargs):
        super().__init__(content, **kwargs)
        self.path = path

    def generate_figure(self, name, label):
        self.content += f"""

        \\begin{{figure}}[ht]
        \\begin{{center}}
        \\scalebox{{0.4}}{{
           \\includegraphics{{{self.path}}}
        }}

        \\caption{{
        \\label{{{label}}}
             {name}.}}
        \\end {{center}}
        \\end {{figure}}

        """


class LatexTableGenerator(LatexGenerator):
    sep = '\\hline\n'
    end = '\\\\\n'

    def __init__(self, content, **kwargs):
        super().__init__(content, **kwargs)

    def df_to_latex(self, df: pd.DataFrame, caption):
        body = ''
        columns_width = [1.5 for i in range(df.index.nlevels)] + [1.5 for i in range(len(df.columns))]

        body += self.generate_columns(df.index.names, df.columns)
        for name, row in df.iterrows():
            row = self.generate_row(name, row)
            body += row
        body += '\\hline\n'
        return self.generate_table_header(body, caption, columns_width)

    def generate_table_header(self, body, caption, columns_width, label='result_table'):
        if not isinstance(body, str):
            raise ValueError(f'body должен быть строкой, вместо этого {type(body)}')
        width = ''.join([f'|p{{{width}cm}}' for width in columns_width]) + '|'
        self.content += f"""
        \\begin{{center}}
            \\begin{{longtable}}{{{width}}}
            \\caption{{{caption}}}\\label{{{label}}}\\\\
                {body}
            \\end{{longtable}}
        \\end{{center}}
        """

    def generate_row(self, name, row):
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
        return self.sep + name + values + self.end

    def generate_columns(self, index_names, columns):
        columns = list(index_names) + list(columns)
        self.content += '\\hline\n' + ' & '.join([f'\\textbf{{{name}}}' for name in columns]) + '\\\\\n'


