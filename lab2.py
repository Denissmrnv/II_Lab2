import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import pymorphy3
import os
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import binascii

nltk.download('stopwords')
nltk.download('punkt')


def print_list(my_list):
    for line in my_list:
        print(line)


def list2file(mylist, myfile):
    f = open(myfile, 'w')
    for item in mylist:
        f.write('%s\n' % item)
    f.close()


def file2list(filename):
    with open(filename, 'r') as text:
        mylist = text.readlines()

    return mylist


def create_filtered_files(old_dir):
    stop_words = set(stopwords.words('russian'))
    regex = re.compile(r'[^А-Яа-я\s]')
    preprocess_text = []

    i = 1
    # пробегаем по директории
    for filename in os.listdir(old_dir):
        # работаем с i-м файлом
        with open(os.path.join(old_dir, filename), 'r') as f:
            while True:
                text_line = f.readline()
                text_line = regex.sub(' ', text_line)
                if text_line == '':
                    break
                word_tokens = word_tokenize(text_line)
                filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]
                preprocess_text.append(' '.join(filtered_sentence))
                new_f = open(f'texts/0 ({i}).txt', 'w')
                for inner in preprocess_text:
                    # создаем новые файлы, не содержащие латинских букв, цифр и др. символов по регулярке
                    # а также не содержащие стоп-слова
                    new_f.write(inner.lower() + ' ')
            i += 1
            preprocess_text.clear()
            new_f.close()


def bag_of_words(directory):
    count = 0
    dataset: list[str] = []

    morph = pymorphy3.MorphAnalyzer()

    for file in os.listdir(directory):
        with open(os.path.join(directory, file), 'r') as f:
            text = ''
            for line in f:
                text += line

            text_tokens = word_tokenize(text)
            bow_no_stops = Counter(text_tokens)

            lemmatized_text = [morph.parse(t)[0].normal_form for t in text_tokens]
            bow_lemmatized = Counter(lemmatized_text)
            count += 1
            s = ' '.join(lemmatized_text)
            dataset.append(s)

            print(str(count) + ' | ' + str(bow_no_stops.most_common(5)) + " - " + str(bow_lemmatized.most_common(5)))
   # list2file(dataset, 'test_dir2/f1_norm.txt')

    return dataset


def tf_idf(dataset):
    tf_idf_transformer = TfidfTransformer(use_idf=True)
    count_vectorizer = CountVectorizer()
    word_count = count_vectorizer.fit_transform(dataset)
    new_tf_idf = tf_idf_transformer.fit_transform(word_count)
    df = pd.DataFrame(new_tf_idf[1].T.todense(), index=count_vectorizer.get_feature_names_out(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    print(df.head(5))
    print(df.size)


# 6. Алгоритм шинглов
# функция для разбиения текста на шинглы
def shingle_text(dir, shingle_size):
    dataset = bag_of_words(dir)
    # создание списка шинглов
    shingles = []

    # разбиение текста на слова
    for sequence in dataset:
        words = sequence.split()
        # цикл по словам с формированием шинглов
        for w in range(len(words) - shingle_size + 1):
            shingle = ' '.join(words[w:w + shingle_size])
            shingles.append(shingle)
    # возврат списка шинглов
    return shingles


# функция для вычисления хэшей шинглов
def hash_shingles(shingles):
    # создание списка хэшей шинглов
    hashes = []
    # цикл по шинглам с вычислением хэшей
    for shingle in shingles:
        # преобразование шингла в байтовую строку
        shingle_bytes = shingle.encode('utf-8')
        # вычисление хэша с помощью функции crc32
        shingle_hash = binascii.crc32(shingle_bytes)
        # добавление хэша в список
        hashes.append(shingle_hash)
    # возврат списка хэшей
    return hashes


# функция для сравнения двух текстов по хэшам шинглов
def compare_texts(dir1, dir2, shingle_size):
    # разбиение текстов на шинглы
    shingles1 = shingle_text(dir1, shingle_size)
    shingles2 = shingle_text(dir2, shingle_size)
    # вычисление хэшей шинглов
    hashes1 = hash_shingles(shingles1)
    hashes2 = hash_shingles(shingles2)
    # вычисление количества одинаковых хэшей
    common_hashes = set(hashes1).intersection(set(hashes2))
    similarity = len(common_hashes) / (len(hashes1) + len(hashes2) - len(common_hashes))
    # возврат результата
    return similarity


if __name__ == '__main__':
    # create_filtered_files('texts')
    tf_idf(bag_of_words('texts'))
    similarity = compare_texts('test_dir1', 'test_dir2', 4)
    print('Result:', similarity*100)
