from transformers import pipeline

#
# Pipelines:
# feature-extraction (get the vector representation of a text)
# fill-mask
# ner (named entity recognition)
# question-answering
# sentiment-analysis
# summarization
# text-generation
# translation
# zero-shot-classification
#


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'sentiment-analysis')
classifier = pipeline("sentiment-analysis")
result = classifier(["I've been waiting for a HuggingFace course my whole life.",
                     "I hate this so much!"])
print(result)
# [{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'zero-shot-classification')
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"]
)
print(result)
# {'sequence': 'This is a course about the Transformers library', 'labels': ['education', 'business', 'politics'], 'scores': [0.8445988297462463, 0.11197440326213837, 0.04342682659626007]}

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'text-generation')
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to")
print(result)
# [{'generated_text': "In this course, we will teach you how to work with your brain to create new problems, and how to take them to the next level and live it out for years and decades longer.\n\nDo you have a problem you know that can't"}]

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'text-generation distilgpt2')
generator = pipeline("text-generation", model="distilgpt2")
result = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(result)
# [{'generated_text': 'In this course, we will teach you how to code the new programming language, learn how to code it, and learn how to use it to make'}, {'generated_text': 'In this course, we will teach you how to use the builtins to do things for yourself from the comfort of your PC, tablet and computer.'}]

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'fill-mask')
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
print(result)
# [{'score': 0.19619810581207275, 'token': 30412, 'token_str': ' mathematical', 'sequence': 'This course will teach you all about mathematical models.'}, {'score': 0.04052736610174179, 'token': 38163, 'token_str': ' computational', 'sequence': 'This course will teach you all about computational models.'}]

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'ner')
ner = pipeline("ner", grouped_entities=True)
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)
# [{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18}, {'entity_group': 'ORG', 'score': 0.9796019, 'word': 'Hugging Face', 'start': 33, 'end': 45}, {'entity_group': 'LOC', 'score': 0.9932106, 'word': 'Brooklyn', 'start': 49, 'end': 57}]

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'question-answering')
question_answerer = pipeline("question-answering")
result = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn"
)
print(result)
# {'score': 0.6949766278266907, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'summarization')
summarizer = pipeline("summarization")
result = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
print(result)
# [{'summary_text': ' America has changed dramatically during recent years . The number of engineering graduates in the U.S. has declined in traditional engineering disciplines such as mechanical, civil,    electrical, chemical, and aeronautical engineering . Rapidly developing economies such as China and India continue to encourage and advance the teaching of engineering .'}]

print()


# ------------------------------------------------------------------------------------------

print('-------------------------------------- %s --------------------------------------' % 'Helsinki-NLP/opus-mt-fr-en')
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
print(result)
# [{'translation_text': 'This course is produced by Hugging Face.'}]

print()


