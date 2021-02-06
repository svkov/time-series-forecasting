rule clean_all:
    shell: 'del /q/f/s reports\\* data\\interim\\*.* data\\processed\\*.* data\\raw\\*.* > NUL'

rule clean_trade:
    shell: 'del /q/f/s reports\\figures_trade_accuracy reports\\*.*' # TODO: