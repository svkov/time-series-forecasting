import pandas as pd


class LatexGenerator:
    content = ''

    def __init__(self, content='', **kwargs):
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
        try:
            parts = [str(part) for part in parts]
        except ValueError as e:
            raise ValueError(f'Ошибка при конкатенации латеха: {e}')

        self.content += '\n'.join(parts)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.content)

    def clear(self):
        self.content = ''

    def __repr__(self):
        return self.content


class LatexPictureGenerator(LatexGenerator):

    def __init__(self, content='', path=None, name=None, label=None, windows_path=True, scale=0.6, **kwargs):
        super().__init__(content, **kwargs)
        self.path = path
        self.name = name
        self.label = label
        self.scale = scale

        if windows_path:
            self._windows_path_to_tex()
        self._process_path()

        self.generate_figure()

    def generate_figure(self):
        self.content += f"""

        \\begin{{figure}}[ht]
        \\begin{{center}}
        \\scalebox{{{self.scale}}}{{
           \\includegraphics{{{self.path}}}
        }}

        \\caption{{
        \\label{{{self.label}}}
             {self.name}.}}
        \\end {{center}}
        \\end {{figure}}

        """

    def _process_path(self):
        self.path = f'../{self.path}'

    def _windows_path_to_tex(self):
        self.path = self.path.replace('\\', '/')


class LatexTableGenerator(LatexGenerator):
    end_separator = '\\hline\n'

    def __init__(self, content='', path=None, caption='', label='', index_cell_width=1.5, columns_cell_width=1.5, **kwargs):
        super().__init__(content)
        self.index_cell_width = index_cell_width
        self.columns_cell_width = columns_cell_width

        if path is not None:
            self.path = path
            self.caption = caption
            self.label = label
            self.df = None
            self._read_csv(**kwargs)
            print(self.df)
            self.append_df_to_content()

    def _read_csv(self, **kwargs):
        self.df = pd.read_csv(self.path, **kwargs)

    def append_df_to_content(self):
        self.df_to_latex(self.df, self.caption, self.label)

    def df_to_latex(self, df: pd.DataFrame, caption, label):
        columns_width = self._get_columns_width(df)
        self._generate_columns(df)
        self._generate_table_content(df)
        self._wrap_table_with_header(caption, columns_width, label)

    def _get_columns_width(self, df: pd.DataFrame):
        index_width = [self.index_cell_width for _ in range(df.index.nlevels)]
        columns_width = [self.columns_cell_width for _ in range(len(df.columns))]
        return index_width + columns_width

    def _generate_columns(self, df: pd.DataFrame):
        index_names = df.index.names
        columns = df.columns
        columns = list(index_names) + list(columns)
        self.content += '\\hline\n' + ' & '.join([f'\\textbf{{{name}}}' for name in columns]) + '\\\\\n'

    def _generate_table_content(self, df: pd.DataFrame):
        rows = [LatexTableRow(name=name, row=row) for name, row in df.iterrows()]
        self.concat_parts(*rows)
        self._add_end_separator()

    def _add_end_separator(self):
        self.content += self.end_separator

    def _wrap_table_with_header(self, caption, columns_width, label):
        width = ''.join([f'|p{{{width}cm}}' for width in columns_width]) + '|'
        self.content = f"""
        \\begin{{center}}
            \\begin{{longtable}}{{{width}}}
            \\caption{{{caption}}}\\label{{{label}}}\\\\
                {self.content}
            \\end{{longtable}}
        \\end{{center}}
        """


class LatexTableRow(LatexGenerator):
    sep = '\\hline\n'
    end = '\\\\\n'

    def __init__(self, name, row, **kwargs):
        super().__init__(content='', **kwargs)
        self.name = name
        self.row = row

        self._parse_index_name()
        self._parse_row()

        self._generate_row()

    def _parse_index_name(self):
        if isinstance(self.name, list) or isinstance(self.name, tuple):
            names = list(self.name)
            self.name = ' & '.join(names) + ' & '
        else:
            self.name = f'{self.name} & '

    def _parse_row(self):
        if not isinstance(self.row, list):
            self.row = self.row.tolist()
        self.row = ' & '.join(map(self._process_row_value, self.row))

    def _process_row_value(self, x):
        if isinstance(x, float):
            return str(round(x, 1))
        return str(x)

    def _generate_row(self):
        self.content = self.sep + self.name + self.row + self.end


if __name__ == '__main__':
    df = pd.DataFrame({'a': 1, 'b': 2}, index=[1, 2])
    lg_table = LatexTableGenerator()
    print(lg_table.content)
    lg_table.df_to_latex(df, 'abc', 'abc')
    lg_table.save('test.tex')

    lg_picture = LatexPictureGenerator(path='dag.svg.png')
    lg_picture.save('text_picture.tex')