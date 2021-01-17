
import tkinter as tk

import sys

sys.path.append(r"F:\NUS-NLP Project")
import get_department
import get_sentiment
import get_gender_age
import get_similarity
import get_topic



#in_string="this is a great toy"

# out_string=get_department.return_category(in_string)
# print(out_string)

# out_emotion=get_sentiment.return_sentiment(in_string)
# print(out_emotion)

# gender=get_gender_age.return_gender(in_string)
# print(gender)

root=tk.Tk()

text = tk.Text(root)
text.insert(tk.INSERT, "WELCOME USER")
text.pack()

def get_category(string):
    out_string=get_department.return_category(string)
    return out_string

def get_emotion(string):
    out_emotion=get_sentiment.return_sentiment(string)
    return out_emotion

def get_gender(string):
    out_gender=get_gender_age.return_gender(string)
    return out_gender

def out_age(string):
    out_age=get_gender_age.return_age_group(string)
    return out_age

def get_similar(string):
    out_similar=get_similarity.return_similar(string)
    return out_similar

def get_label(string):
    out_label=get_topic.review_label(string)
    return out_label


def ifclick():

    #text.delete('1.0', END)
    text.insert(tk.INSERT, "\n\n")
    text.insert(tk.INSERT, " ENTER REVIEW \n")
    #text.insert(tk.INSERT, " CURRENT CATEGORY IS "+ product_category)
    s1=entry1.get()
    text.insert(tk.INSERT, "USER:" +s1)
    text.insert(tk.INSERT, "\n")

    product_category=get_category(s1)
    emotion=get_emotion(s1)
    gender=get_gender(s1)
    age=out_age(s1)
    similar=get_similar(s1)
    labels=get_label(s1)
    text.insert(tk.INSERT, "AGENT: Suggested Product Category is "+ product_category + " \n")
    text.insert(tk.INSERT, "\nSuggested Emotion  is "+ emotion + " \n")
    text.insert(tk.INSERT, "\nSuggested Gender  is "+ gender + " \n")
    text.insert(tk.INSERT, "\nSuggested Age  is "+ age + " \n")
    text.insert(tk.INSERT, "\nSuggested Similar Review is " + similar + "\n")
    text.insert(tk.INSERT, "\nSuggested Topics are "+ labels + "\n" )
    
    text.insert(tk.INSERT, "\n")
    text.insert(tk.END, "NEXT COMMAND ")
    text.pack()

entry1=tk.Entry(root, width="100")

button=tk.Button(root, text="CLICK", command=ifclick)

entry1.pack()
button.pack()
root.mainloop()
