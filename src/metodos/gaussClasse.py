from typing import List


class MetodoGauss:
    """
    Essa classe realiza todos os métodos necessáiros para resolver o sistema através do metodo de GAUSS.
    """

    def __init__(self):
        self.resultados_sistemas = []
        self.resultados_tempo = []

    def calcula_matriz_triangular_inferior(
        self,
        ordem_matriz: int,
        matriz_A: List[List[float]],
        vetores_b: List[float],
    ) -> None:
        """
        Essa função realiza calculos para obter uma matriz triangular inferior e passar essa matriz para  a funcao realiza_substituicao
        Essa função também chama a função realiza_privoteamento para diminui os erros de arredondamento e divisão por zero.
        Objetivo: Obter uma matriz triangular inferior

        :param
           - ordem_matriz: a ordem da matriz.
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - vetores_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
        :return - vazio

        """
        lista_M = []

        for etapas in range(0, ordem_matriz - 1):
            self.realiza_pivoteamento(matriz_A, etapas, ordem_matriz, vetores_b)
            for linhas in range(etapas + 1, ordem_matriz):
                lista_M.append(
                    matriz_A[linhas][etapas] / matriz_A[etapas][etapas]
                )
                for colunas in range(0, ordem_matriz):
                    matriz_A[linhas][colunas] = matriz_A[linhas][colunas] - (
                        lista_M[0] * matriz_A[etapas][colunas]
                    )
                vetores_b[linhas] = vetores_b[linhas] - (
                    lista_M[0] * vetores_b[etapas]
                )
                lista_M.clear()

        self.realiza_substituicao(matriz_A.copy(), vetores_b, ordem_matriz)

    def realiza_pivoteamento(
        self,
        matriz_A: List[List[float]],
        etapas: int,
        ordem_matriz: int,
        vetores_b: List[float],
    ) -> None:
        """
        Essa função realiza o pivoteamento de uma matriz_A e dos vetores_b.Ela verifica se o pivo é o maior número da coluna
        Caso seja o maior, ela apenas retorna.
        Caso o pivo não seja o maior, ela troca as linhas.
        Objetivo: Minimizar os erros de arredondamento e a divisão por zero.

        :param
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - etapas: quantidade de etapas que deverao ser executadas, isso é decidido fazendo: ordem_matriz-1
           - ordem_matriz: a ordem da matriz do sistema.
           - vetores_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
        :return - vazio

        """
        maior_numero = abs(matriz_A[etapas][etapas])
        pivo_atual = True
        linha_maior = 0

        for i in range(etapas, ordem_matriz):
            if abs(matriz_A[i][etapas]) > maior_numero:
                maior_numero = abs(matriz_A[i][etapas])
                pivo_atual = False
                linha_maior = i

        if pivo_atual == False:
            matriz_A[linha_maior], matriz_A[etapas] = (
                matriz_A[etapas].copy(),
                matriz_A[linha_maior].copy(),
            )
            vetores_b[linha_maior], vetores_b[etapas] = (
                vetores_b[etapas].copy(),
                vetores_b[linha_maior].copy(),
            )
        else:
            return

    def realiza_substituicao(
        self,
        matriz_A: List[List[float]],
        vetores_b: List[float],
        ordem_matriz: int,
    ) -> None:
        """
        Essa função recebe uma matriz inferior triangular e realiza a substituição sucessiva para descobrir
        o valor das  variáveis do sistema que foi passado.
        Objetivo: Descobrir as variáveis do sistema.

        :param
           - matriz_A: dados com a matriz A fornececida pelo arquivo passado como argumento.
           - vetores_b: dados fornecidos pelo arquivo passado como argumento que sao os vetores b do sistema.
           - ordem_matriz: a ordem da matriz do sistema.
        :return - vazio

        """

        variaveis_resolvidas = [0] * int(ordem_matriz)

        variaveis_resolvidas[ordem_matriz - 1] = (
            vetores_b[ordem_matriz - 1]
        ) / (matriz_A[ordem_matriz - 1][ordem_matriz - 1])

        for i in range(ordem_matriz - 2, -1, -1):
            soma = 0
            for j in range(i + 1, ordem_matriz):
                soma += matriz_A[i][j] * variaveis_resolvidas[j]
            variaveis_resolvidas[i] = (vetores_b[i] - soma) / matriz_A[i][i]

        self.resultados_sistemas.append(variaveis_resolvidas)

    def imprime_resultados_gauss(self, qtd_sistemas: int) -> None:
        """
        Essa função imprime o resultado dos sistemas que foram passados bem como o tempo de execução de cada sistema.
        É válido destacar que a primeira posicao do vetor resultados_sistemas representaria a variável X e assim por
        diante.
        Por exemplo, em um sistema 3x3 nos teriamos: variavel x = resultados_sistema[0], variavel y = resultados_sistemas[1]
        e por ultimo a variavel z = resultados_sistemas[2]
        Objetivo: imprimir resultados do sistema e tempo de execução.

        :param
           - qtd_sistemas: quantidade de sistemas no arquivo que foi fornececido.
        :return - vazio

        """

        print("Resultados dos sistemas resolvidos pelo Metodo de GAUSS:")
        print(
            f"Resultado para o sistema {qtd_sistemas+1}: {self.resultados_sistemas[qtd_sistemas]} Tempo: {self.resultados_tempo[qtd_sistemas]} segundos"
        )
        print()

    def add_resultados_tempo(self, tempo: int) -> None:
        """
        Essa função adiciona o tempo de execução a lista de resultados_tempo.
        Objetivo: adicionar tempo de execução

        :param
           - tempo: tempo gasto para executar cada sistema.
        :return - vazio

        """
        self.resultados_tempo.append(tempo)

    def get_resultados(self) -> List[float]:
        """
        Função usada para os testes dessa classe.

        :return - Uma lista contendo os resultados dos sitemas.

        """
        return self.resultados_sistemas
