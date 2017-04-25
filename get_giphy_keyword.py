#pattern patt=re.compile('(?i)<meta name="keywords" content="(.*?)GIF, Animated GIF">')


import re
import urllib2
import giphypop


#build the pattern

def get_gif():
	g = giphypop.Giphy()
	results = g.search_list('happy',limit = 100)

	for i in range(len(results)):
		gif_url = results[i].url
		gif_image_url = results[i].media_url
		get_giphy_keywords(gif_url,gif_image_url)

def get_giphy_keywords(url,imgURL):
	patt=re.compile('(?i)<meta name="keywords" content="(.*?)GIF, Animated GIF">')
	#read str info from url
	web_info=urllib2.urlopen(url).read()

	#locate the key words
	keywords=re.findall(patt,web_info)[0].split()

	#url directly to gif image
	keywords.insert(0,imgURL)
	print keywords

def main():
	get_gif()
	

if __name__ == "__main__":
    main()
