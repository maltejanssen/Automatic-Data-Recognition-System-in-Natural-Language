# Annotations of WNUT'17 Emerging and Rare Entities task
origin: https://github.com/leondz/emerging_entities_17

## General Info
The Dataset is focused on merging  and rare entities and contains very few surface forms which mostly don't repeat more than once. Most of the surface forms also are not shared by Training and Test Data.

### Training Data
The training Data contains 1,000  annotated tweets, totaling 65,124 tokens.

### Test Data
The test Data not only contains tweets, but also posts from Reddit, Youtube and Stackexchange. The reason for that is to have posts with more than 140 characters as these exhibit different writing styles and characteristics.

### Classes
The data is divided into the following 7 classes:

1. Person
2. Location (including GPE, facility)
3. Corporation 
4. Product (tangible goods, or well-defined services)
5. Creative work (song, movie, book and so on)
6. Group (subsuming music band, sports team,and non-corporate organisations)
7. O

Guidelines of Classes (see: http://aclweb.org/anthology/W17-4418): 

## 1. Person 
instances in training Data: 995
instances in testing Data: 532

Percentage of training Data: ~1,6%
Percentage of testing Data: ~2,3%

Names of people (e.g. **Virginia Wade**). Don't mark people that don't have their own name. Include punctuation in the middle of names. Fictional people can be included, as long as they're referred to by name (e.g. "Harry Potter").


## 2. Location 
instances in training Data: 793
instances in testing Data: 168

Percentage of training Data: ~1,3%
Percentage of testing Data:  ~0,8%

Names that are locations (e.g. **France**). Don't mark locations that don't have their own name. Include punctuation in the middle of names. Fictional locations can be included, as long as they're referred to by name (e.g. "Hogwarts").

## 3. Corporation
instances in training Data: 267
instances in testing Data: 192

Percentage of training Data: ~0,4%
Percentage of testing Data: ~0,8%

Names  of  corporations  (e.g.Google).   Don’t  mark  locations  that  don’t  havetheir own name. Include punctuation in the middleof names

## 4. Product
instances in training Data: 345
instances in testing Data: 198

Percentage of training Data: ~0,5%
Percentage of testing Data: ~0,8%

Name of products (e.g. **iPhone**). Don't mark products that don't have their own name. Include punctuation in the middle of names.

There may be no products mentioned by name in the sentence at all - that's OK. Fictional products can be included, as long as they're referred to by name (e.g. "Everlasting Gobstopper"). It's got to be something you can touch, and it's got to be the official name.


## 5. Creative work
instances in training Data: 346
instances in testing Data: 468

Percentage of training Data: ~0,6%
Percentage of testing Data: ~2,0%

Names of creative works (e.g. **Bohemian Rhapsody**). Include punctuation in the middle of names. The work should be created by a human, and referred to by its specific name.



## 6. Group
instances in training Data: 414
instances in testing Data: 241

Percentage of training Data: ~0,7%
Percentage of testing Data: ~1,0%

Names of groups (e.g. **Nirvana**, **San Diego Padres**). Don't mark groups that don't have a specific, unique name, or companies.

There may be no groups mentioned by name in the sentence at all - that's OK. Fictional groups can be included, as long as they're referred to by name.


## 7. O
instances in training Data: 59570
instances in testing Data: 21934

Percentage of training Data: ~94,7%
Percentage of testing Data: ~93,8%

Anything that doesn't fit any of the above


