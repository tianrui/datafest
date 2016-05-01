import newspaper
import numpy as np
import pdb
import urllib
import json

from scrapy.spiders             import BaseSpider
from scrapy.selector        import HtmlXPathSelector, Selector
#from nettuts.items          import NettutsItem
from scrapy.http            import Request
from HTMLParser import HTMLParser

import argparse
import re

class SmallSpider(BaseSpider):
    name            = "ArticleSpider"
    allowed_domains = ["*"]
    start_urls      = ["https://www.google.ca/#hl=en&gl=ca&tbm=nws&authuser=0&q="]

    def __init__(self, keyword, name="ArticleSpider", allowed_domain="*"):
        self.name=name
        self.allowed_domains=allowed_domain
        self.start_urls="https://www.google.ca/#hl=en&gl=ca&tbm=nws&authuser=0&q=" + "+".join(keyword.split(" "))
        return

    def parse(self, response):
        hxs             = Selector(response)
        links           = hxs.xpath("//a/@href").extract()

        #We stored already crawled links in this list
        crawledLinks    = []

        #Pattern to check proper link
        linkPattern     = re.compile("^(?:ftp|http|https):\/\/(?:[\w\.\-\+]+:{0,1}[\w\.\-\+]*@)?(?:[a-z0-9\-\.]+)(?::[0-9]+)?(?:\/|\/(?:[\w#!:\.\?\+=&amp;%@!\-\/\(\)]+)|\?(?:[\w#!:\.\?\+=&amp;%@!\-\/\(\)]+))?$")

        for link in links:
            # If it is a proper link and is not checked yet, yield it to the Spider
            if linkPattern.match(link) and not link in crawledLinks:
                crawledLinks.append(link)

#                yield Request(link, self.parse)

#        titles  = hxs.select('//h1[@class="post_title"]/a/text()').extract()
#        for title in titles:
#            item            = NettutsItem()
#            item["title"]   = title
#            yield item

class specialParser(HTMLParser):
    def init_links(self):
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    print name, "=", value
                    self.links.append(value)
        return


def search_article(keywords):
    """ Search for articles with keyword
    """
    query = urllib.urlencode({'q': keywords})
    url = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&' + query
    #url = 'https://www.google.ca/#hl=en&gl=ca&tbm=nws&authuser=0&q=' + '+'.join(keywords.split(' ')) + '&ei=bqolV-_pFsaye4uVmIgC'
    print 'Search URL: ', url

    response = urllib.urlopen(url).read()
    resp_json = json.loads(response)
    results = resp_json['responseData']['results']
    article_sources = []
    for result in results:
        title = result['title']
        resurl = result['url']
        print ( title + ': ' + resurl )
        article_sources.append(resurl)


    return article_sources
    

def extend_adwords(keywords, article_sources):
    """ Extend the keywords for an event based on searched articles
    """
    o_keys = list(set(keywords.split(" ")))
    n_keys = set()
    for newsource_url in article_sources:
        newsource = newspaper.build(newsource_url, language='en')
        for article in newsource.articles[:10]:
            #article = newspaper.Article(url=link, language='en')
            article.download()
            if article.html == '':
                print "Unable to download"
            else:
                article.parse()
                print article.text
                article.nlp()
                n_keys.extend([word.encode('ascii', 'ignore') for word in article.keywords])
                print "Article keywords: ", article.keywords
    pdb.set_trace()
    n_keys.extend(o_keys)
    o_keys = list(set(n_keys))
    #print "Overall new keywords: ", o_keys
    return o_keys
            

def arg_parse():
    parser = argparse.ArgumentParser(description='Extending adwords on Ticketmaster')
    parser.add_argument('-keywords', default="", type=str,
                        help='keywords in a string delimited by space')

    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    links = search_article(args.keywords)
    new_keys = extend_adwords(args.keywords, links)
    print "Augmented keys: ", new_keys

    return 

if __name__=='__main__':
    main()
