#coding:utf-8
import feedparser
ny=feedparser.parse("http://newyork.craigslist.org/stp/index.rss")
print(ny['entries'])
print(len(ny['entries']))