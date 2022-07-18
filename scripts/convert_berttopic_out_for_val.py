from dataclasses import field
import pandas as pd

def improve_kw_readabiliity(keyword):
    """
    This function improves readability of keywords output from BERTopic.
    For instance, converting the following:

    [('my', 0.014973503424716761), ('and', 0.011889245058485058), ('to', 0.011812021745623659), \
        ('keto', 0.011288345943019397), ('on', 0.009979384479349658), ('was', 0.00960507459455274), \
            ('the', 0.009495719292829143), ('it', 0.00937330531711953), ('for', 0.008818634802585984), ('me', 0.008780988898534944)]
    
    to the following:

    'my_and_to_keto_on_was_the_it_for_me'.

    """

    return "_".join(keyword.split('\'')[1::2])


if __name__ == '__main__':
    
    df = pd.read_csv(input('Please enter name of CSV file output from BERTopic.'))
    
    # Shuffle
    df = df.sample(frac=1)

    # Change keyword notation so it's more readable.
    df['keywords'] = df['keywords'].map(improve_kw_readabiliity)

    # Now see by which field we want to sample from.
    field_of_interest = input('Which field do you want to sample groups from?')
    num_samples = int(input('How many samples do you want?'))

    df.groupby(field_of_interest).head(num_samples).to_csv("/Users/ssomani/Desktop/df_{0}_sampled.csv".format(field_of_interest))