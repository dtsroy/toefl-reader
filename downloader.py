import json

import requests
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import os
import re
import threading

dest = './data/raw/'
cache = './data/cache/'


class Downloader:
    LABELS = 'ABCDEF'

    Q_13 = 'Look at the four squares [■] to insert the sentence in the passage.'
    Q_14 = ('Directions: An introductory sentence for a brief summary of the passage is provided below. Complete the '
            'summary by selecting THREE answer choices that express the most important ideas in the passage. Some '
            'answer choices do not belong in summary because the express ideas that are not presented in the passage '
            'or are minor ideas in the passage.')

    def __init__(self, cache, raw):
        self.cache = cache
        self.raw = raw

        self.error_count = 0

    @classmethod
    def _get_homepage_list(cls):
        """
        从首页获取套题链接
        """
        url = 'https://top.zhan.com/toefl/read/alltpo.html'
        bsp = bs(requests.get(url).text, 'lxml')
        return (
            item['href']
            for item in bsp.find_all('ul',
                                     class_='cssTopTitleList clearfix')[0]
        .find_all('a')
        )

    @classmethod
    def _get_article_list(cls, url):
        """
        获取一页下的文章列表
        """
        bsp = bs(requests.get(url).text, 'lxml')
        r = []
        for div in bsp.find_all('div', class_='cssExeBody'):
            for div_ in div.find_all('div', class_='cssMask'):
                r.append(div_.find_all('a')[3]['href'])
        return r

    @classmethod
    def _get_questions(cls, url):
        """
        获取各个题目的链接
        """
        bsp = bs(requests.get(url).text, 'lxml')
        questions = []
        for a in bsp.find_all('div', id='footer_review')[0].find_all('a'):
            questions.append(a['href'])
        # print(questions)
        return questions

    @classmethod
    def _get_article(cls, url):
        """
        获取文章本身
        !!!必须使用13题的页面，才包含填充句子的标记!!!
        """
        bsp = bs(requests.get(url).text, 'lxml')
        div = bsp.find('div', class_='article')
        for br in div.find_all('br'):
            br.replace_with('\n')
        return div.get_text().replace('\n\n', '\n')[1:]

    @classmethod
    def _get_question_general(cls, url):
        """
        获取普通题目
        """
        try:
            bsp = bs(requests.get(url).text, 'lxml')
            qs = bsp.find('div', class_='question_option')
            q = re.sub(r'^\d{1,2}[.]', '', qs.find('div', class_='q_tit').get_text()[1:-3])
            options = []

            for p in qs.find_all('p'):
                options.append(p.get_text()[5:-1])

            correct = cls.LABELS.index(bsp.find('div', class_='left correctAnswer').find('span').get_text())

            return q, options, correct
        except Exception as e:
            print(e, url)
            return None

    @classmethod
    def _get_question_13(cls, url):
        try:
            bsp = bs(requests.get(url).text, 'lxml')
            div = bsp.find('div', class_='question_option')
            return (
                # div.find('div', class_='left text').get_text()[3:-1],
                # 'Look at the four squares [■] to insert the sentence in the passage.',
                div.find('p').get_text()[21:-19],
                cls.LABELS.index(bsp.find('div', class_='left correctAnswer').find('span').get_text())
            )
        except Exception as e:
            print(e, url)
            return None

    @classmethod
    def _get_question_14(cls, url):
        try:
            bsp = bs(requests.get(url).text, 'lxml')
            div = bsp.find('div', class_='question_option margpadding')
            options = []
            for p in div.find_all('p'):
                options.append(p.get_text()[23:-18])
            cs = [cls.LABELS.index(a)
                  for a in bsp.find('div', class_='left correctAnswer').find('span').get_text()]
            return options, cs
        except Exception as e:
            print(e, url)
            return None

    def cache_article_list(self):
        a = []
        for h in tqdm(self._get_homepage_list()):
            for article in self._get_article_list(h):
                a.append(article)
        with open(os.path.join(self.cache, 'article_list.json'), 'w+') as f:
            json.dump(a, f, indent=4)

    def _download_range(self, lst, start):
        tmp = {}
        flag = False
        for ai, i in enumerate(lst):
            qs = self._get_questions(i)
            tmp['article'] = self._get_article(qs[-2])
            tmp['questions'] = []
            for idx, q in enumerate(qs):
                if idx == len(qs) - 2:
                    # 第十三题
                    tmp['questions'].append(q := self._get_question_13(q))

                elif idx == len(qs) - 1:
                    # 第十四题
                    tmp['questions'].append(q := self._get_question_14(q))
                else:
                    tmp['questions'].append(q := self._get_question_general(q))
                if not q:
                    flag = True
                    print(f'start={start}, ai={ai}, idx={idx}')
                    break
            if flag:
                # 出现问题了
                flag = False
                self.error_count += 1
                continue
            with open(os.path.join(self.raw, '%d.json' % (ai + start)), 'w+') as f:
                json.dump(tmp, f, indent=4)

    def download_t(self, task_per_threads):
        with open(os.path.join(self.cache, 'article_list.json'), 'r') as f:
            articles = json.load(f)
        n = len(articles) // task_per_threads
        print(n, len(articles) % task_per_threads)
        # return
        for i in range(n):
            threading.Thread(
                target=self._download_range,
                args=(articles[i * task_per_threads:i * task_per_threads + task_per_threads], i * task_per_threads)
            ).start()
        if len(articles) % task_per_threads != 0:
            threading.Thread(
                target=self._download_range,
                args=(articles[n * task_per_threads:], n * task_per_threads)
            ).start()

    def download(self):
        with open(os.path.join(self.cache, 'article_list.json'), 'r') as f:
            articles = json.load(f)
        self._download_range(articles, 0)


d = Downloader(cache, dest)
# print(Downloader._get_question_general('https://top.zhan.com/toefl/read/practicereview-951-13-0-10.html'))
# print(Downloader.cache_article_list(cache))
# d.download()
d.download_t(15)
