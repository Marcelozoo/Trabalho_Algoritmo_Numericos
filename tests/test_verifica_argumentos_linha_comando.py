from os import path
import sys
import os
import pytest

diretorio_projeto_atual = path.dirname(__file__)
diretorio_projeto_atual = path.join("..")
sys.path.append(diretorio_projeto_atual)


diretorio_projeto_atual = os.path.join(os.path.dirname(__file__), "..")
diretorio_projeto_atual = os.path.normpath(diretorio_projeto_atual)
sys.path.append(diretorio_projeto_atual)

from src.verifica_argumentos_linha_comando import verifica_linha_de_comando
from src.ler_arquivo import ler_arquivo


def test_verifica_argumento_linha_comando():
    """
    Testa quantidade de argumentos na linha de comando
    """

    sys.argv[0] = "src/main.py"
    sys.argv[1] = "arquivos/arq.txt"

    assert len(sys.argv) == 2
    assert verifica_linha_de_comando() == "arquivos/arq.txt"

    sys.argv = ["src/main.py"]
    assert len(sys.argv) == 1

    sys.argv = ["src/main.py", "arquivos/arq.txt", "argumento2"]
    assert len(sys.argv) >= 3


def test_verifica_arquivo_nao_encontrado():
    """
    Testa se uma exceção foi lançada nesse caso a FileNotFoudError.
    Caso ela seja lançada o teste é passado.
    """
    with pytest.raises(FileNotFoundError):
        caminho_arquivo = "arquivos/inexistente.txt"
        ler_arquivo(caminho_arquivo)
