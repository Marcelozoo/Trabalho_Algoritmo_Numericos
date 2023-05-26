import sys
from gerenciaLinhaDeComando.verifica_argumentos_linha_comando import (
    verifica_linha_de_comando,
)
from gerenciadorDeArquivos.ler_arquivo import ler_arquivo
from timeit import timeit
from metodos.gaussClasse import MetodoGauss
from metodos.LuClasse import MetodoLu
from metodos.gaussJacobiClass import MetodoGaussJacobi
from metodos.gaussSeidelClass import MetodoGaussSeidel


"""
    GRUPO: MARCELO BENTO CÔGO
    
    Arquivo principal que roda o programa.
    Primeiramente ele executa a chamada da função linha_de_comando e verifica se os argumentos foram definidos corretamente.
    Caso o  argumento do arquivo nao exista ele captura a exception e imprime uma mensagem de erro e o programa fecha.
    Caso o número de argumentos seja maior ou menor do que o necessário o programa exibe uma mensagem de erro e fecha.
    Após isso ele cria um objeto da classe MetodoGauss,MetodoLu, MetodoGaussJacobi e MetodoGaussSeidel.
    Com isso os calculos para os sistemas começam.

        
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

    metodo_gauss = MetodoGauss()
    metodo_lu = MetodoLu()
    metodo_gauss_jacobi = MetodoGaussJacobi()
    metodo_gauss_seidel = MetodoGaussSeidel()

    tempo_total_gauss = 0
    tempo_total_lu = 0
    tempo_total_gauss_jacobi = 0
    tempo_total_gauss_seidel = 0

    for i in range(quantidade_sistemas):
        tempo_total_gauss = timeit(
            lambda: metodo_gauss.calcula_matriz_triangular_inferior(
                ordem_da_matriz, matriz_A.copy(), vetores_b[i].copy()
            ),
            number=1,
        )

        tempo_total_lu = timeit(
            lambda: metodo_lu.calcula_matriz_triangular_inferior_e_superior(
                ordem_da_matriz,
                matriz_A.copy(),
                vetores_b[i].copy(),
                False if i == 0 else True,
            ),
            number=1,
        )

        for j in range(0, 2):
            tempo_total_gauss_jacobi = timeit(
                lambda: metodo_gauss_jacobi.calcula_sistema_linear(
                    ordem_da_matriz,
                    matriz_A.copy(),
                    vetores_b[i].copy(),
                    precisao,
                ),
                number=1,
            )
            if metodo_gauss_jacobi.get_converge() == True:
                break
            else:
                tempo_total_gauss_jacobi = 0

        for j in range(0, 2):
            tempo_total_gauss_seidel = timeit(
                lambda: metodo_gauss_seidel.calcula_sistema_linear(
                    ordem_da_matriz,
                    matriz_A.copy(),
                    vetores_b[i].copy(),
                    precisao,
                ),
                number=1,
            )

            if metodo_gauss_seidel.get_converge() == True:
                break
            else:
                tempo_total_gauss_seidel = 0

        metodo_gauss_seidel.set_converge(True)
        metodo_gauss_jacobi.set_converge(True)

        metodo_gauss.add_resultados_tempo(tempo_total_gauss)
        metodo_lu.add_resultados_tempo(tempo_total_lu)
        metodo_gauss_jacobi.add_resultados_tempo(tempo_total_gauss_jacobi)
        metodo_gauss_seidel.add_resultados_tempo(tempo_total_gauss_seidel)

        tempo_total_gauss = 0
        tempo_total_lu = 0
        tempo_total_gauss_jacobi = 0
        tempo_total_gauss_seidel = 0

    for k in range(0, quantidade_sistemas):
        metodo_gauss.imprime_resultados_gauss(k)
        metodo_lu.imprime_resultados_lu(k)
        metodo_gauss_jacobi.imprimir_resultados(k)
        metodo_gauss_seidel.imprimir_resultados(k)
        print()
