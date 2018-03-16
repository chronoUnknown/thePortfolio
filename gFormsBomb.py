# A program that bombs a Google Forms file with a specific (and often a cringeworthy and terrible) response

"""
First it takes in the google forms url (copy from address bar), the header (using the one provided works just
fine) and the data (I'll get to that), then puts them in a POST request (what the requests module is for).

for the form data: Entry IDs are found by right clicking on a question, selecting 'Inspect Element', then finding the name attribute and copying the value (ie find "name='entry.###'" where ### is some number). Another way to find them is to right-click, select 'view source', pressing CTRL-F, then just finding all instances of the phrase 'entry.'. The Entry IDs represent the questions. For the corresponding data, just enter what's on the form (copying and pasting is generally good enough!). Fill out the form, then just run r a bunch.
"""

import requests

url = '<google_forms_url>/formResponse'

user_agent = {'Referer': '<google_forms_url>/viewform', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

form_data = fdata = {'entry.<iD>': 'data', 'draftResponse': [], 'pageHistory': 0}

r = requests.post(url, data=fdata, headers=user_agent)

# Easy way to mass-use r: for i in range(<SOME REALLY FUCKING HUGE NUMBER>): r
# That'll get 'em (lol)

for i in range(1000):
	r
