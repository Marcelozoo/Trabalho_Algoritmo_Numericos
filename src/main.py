from verifica_argumentos_linha_comando import verifica_linha_de_comando
from ler_arquivo import ler_arquivo
import sys


"""
    GRUPO: MARCELO BENTO CÔGO
    
    Arquivo principal que roda o programa.
    Primeiramente ele executa a chamada da função linha_de_comando e verifica 
    se os argumentos foram definidos corretamente.
    Após isso ele chama a função ler_arquivo que verifica se o arquivo existe 
    e retorna a quantidade_sistemas,  ordem_da_matriz, precisao e o sistema.
    Caso o arquivo nao exista ele captura a exception e imprime uma mensagem de erro e o programa fecha.
        
"""
if __name__ == "__main__":
    try:
        caminho_arquivo = verifica_linha_de_comando()
        (
            quantidade_sistemas,
            ordem_da_matriz,
            precisao,
            matriz_A,
            vetores_b,
        ) = ler_arquivo(caminho_arquivo)

    except FileNotFoundError:
        print("ERRO!Arquivo não encontrado!")
        sys.exit(1)
