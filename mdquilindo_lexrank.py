from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


class LexRank:
    def __init__(self, alfa=0.2, error=0.0001, url=None, summary_length=1, index=[]) -> None:
        self.alfa = alfa
        self.error = error
        self.url = url
        self.summary_length=summary_length
        self.index = index
        self.n = 0

    def get_text_summarization(self):
        """Return:
        n: num iterations for the algorithm to converge
        p: vector in status n
        string with text summarization from docs in vector m,
        """
        m = self.preprocessing_data()

        if self.summary_length >= self.n:
            raise Exception('La cantidad de terminos para el resumen supera la cantidad de oraciones')

        m_similary_cosine = self.get_matrix_similary_cosine(m)
        m_umbral = self.get_matrix_umbral(m_similary_cosine)
        m_centrality = self.get_matrix_centrality(m_umbral)
        m_chain_markov = self.get_matrix_with_dumpy_factor(m_centrality)
        n, p = self.power_method(m_chain_markov)
        p = p.sort_values(by='p', ascending=False)

        with open(self.url) as f:
            lines = f.readlines()

        text_sumarization = ''

        for p_i in p.index.to_numpy()[0:self.summary_length]:
            text_sumarization += lines[int(p_i.split(' ')[1])]
        
        return n, p, text_sumarization

    def preprocessing_data(self):
        if not self.url:
            raise Exception('Ingrese una URL')

        with open(self.url) as f:
            lines = f.readlines()

        index = []
        for i in range(len(lines)):
            index.append('DOC ' + str(i))

        if len(self.index) == 0:
            self.index = index

        self.n = len(self.index)
        
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(lines)
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names_out(), index=self.index)
        return df
        
    def get_matrix_similary_cosine(self, m):
        """Return matrix similary cosine
        m: 
        """
        matrix_transicion=cosine_similarity(m, m)
        df = pd.DataFrame(matrix_transicion, columns=self.index, index=self.index)
        
        return df

    def get_matrix_umbral(self, m):
        """Return matrix umbral
        m: 
        """
        matrix_umbral = m >= 0.2
        matrix_umbral = matrix_umbral.astype(int)
        df = pd.DataFrame(matrix_umbral, columns=self.index, index=self.index)
        return df

    def get_matrix_centrality(self, m):
        """Return matrix centrality
        m: matrix umbral
        """
        grades = m.sum(axis=0)
        matrix_transicion = (m/grades).T
        df = pd.DataFrame(matrix_transicion, columns=self.index, index=self.index)
        return df

    def get_matrix_with_dumpy_factor(self, m):
        """Return matrix irreductible and aperiodic
        m: 
        """
        d = self.alfa
        matrix_transicion = (d / self.n) + (1 - d) * m
        df = pd.DataFrame(matrix_transicion, columns=self.index, index=self.index)
        return df

    def power_method(self, m):
        """Resolved matrix transition or Chain Markov power method"""
        error_d = self.error
        error_a = 1
        # vector 1/n, each sentence has same probability
        p_i = np.ones(self.n)/self.n
        t = 0
        while(error_a >= error_d):
            t += 1
            p_i1 = np.dot(m.transpose(), p_i)
            error_a = np.linalg.norm(p_i1 - p_i)
            p_i = p_i1
        df = pd.DataFrame(p_i, columns=['p'], index=self.index)
        return t, df


lexrank = LexRank(url='./data_news.txt', summary_length=4)
t, p, summary = lexrank.get_text_summarization()
print(t)
print(p)
print(summary)


