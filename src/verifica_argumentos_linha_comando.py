import sys
from typing import Union


def verifica_linha_de_comando() -> Union[str, None]:
    """
    Recebendo o input da linha de comando e verificando os argumentos que foram passados.

    :return - Caso os argumentos estejam corretos,retorna uma string
              com o caminho do arquivo ou seu nome.
              Caso os argumentos estejam incorretos,retorna uma mensagem de erro e encerra o programa.
    """
    if len(sys.argv) == 2:
        return sys.argv[1]

    elif len(sys.argv) == 1:
        print(
            "ERRO!Está faltando como argumento o caminho do arquivo na linha de comando!"
        )
        sys.exit(1)

    else:
        print("ERRO!Quantidade de argumentos supera a quantidade limite!")
        sys.exit(1)
