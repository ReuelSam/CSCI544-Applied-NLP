'''
Command to run this code: python HW2-REUELSAMUEL-SAM.py directory_path

directory_path: the path to which the data files (train, dev, test) are stored
                The filenames MUST be: train, dev and test (WITH NO EXTENSIONS)
                The required output files will also be written into this directory
'''

# imports
import numpy as np
import pandas as pd
import sys
import json

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# function to read the data
def read_data(filename, get_vocab = False, separate_sentences = False, replace_unkown = False, vocabulary = None, tags_present = True):
    file = open(filename, 'r')
    lines = file.readlines()
    lines.append("\n")
    data = []                                       # stores each sentence as a list that contains each word as [position_index, word, tag]
    data_df = []                                    # final dataframe
    temp = [['0', '<START>', '<START_TAG>']]        # default entry at the start of every sentence
    if replace_unkown:                              # for reading data to be tested
        vocab_words = set(vocabulary.unique_words.tolist())

    if get_vocab:
        vocab = []                                  # initializing the vocabulary

    for line in lines:
        if line == '\n':                            # if end of sentence
            data.append(temp)
            temp = [['0', '<START>', '<START_TAG>']]        # replace end of sentence with START (indicates START of next sentence)
        else:
            line = line.split('\t')
            if get_vocab:
                vocab.append(line[1])
            line[-1]=line[-1].strip('\n')
            if (replace_unkown) and (line[1] not in vocab_words):       # replace words not in vocabulary with <UNK>
                line[1] = "<UNK>"
            temp.append(line)
        
    if get_vocab:
        vocab = pd.DataFrame(vocab, columns=["words"])          # store vocabulary as a Dataframe

    for data_sample in data:
        data_df.append(pd.DataFrame(data_sample, columns=['position_index', 'word', 'tag']))            # write the data sentence by sentence -> list of dataframes
        if not tags_present:
            data_df[-1] = data_df[-1].drop(columns = 'tag')

    data_df_combined = pd.concat(data_df)                       # collapse each sentence to occur one after the other -> 1 dataframe

    if get_vocab and not separate_sentences:
        return(data_df_combined, vocab)
    if get_vocab and separate_sentences:
        return(data_df, vocab)
    if not get_vocab and not separate_sentences:
        return(data_df_combined)
    else:
        return(data_df)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# function to create the vocabulary
def create_vocab(data, vocab_data, count_threshold=2):
    vocab_no_dupl = vocab_data['words'].value_counts()
    vocab_no_dupl = pd.DataFrame(list(zip(vocab_no_dupl.index.tolist(), vocab_no_dupl.tolist())), columns=['unique_words', 'count'])
    total_words_before = len(vocab_no_dupl)

    words_to_remove = vocab_no_dupl[vocab_no_dupl['count'] < count_threshold]
    total_removed_words = len(words_to_remove)
    vocabulary = vocab_no_dupl.drop(vocab_no_dupl[vocab_no_dupl['count'] < count_threshold].index)

    vocabulary.loc[-1] = ['<UNK>', total_removed_words]  # adding a row
    vocabulary.index = vocabulary.index + 1  # shifting index
    vocabulary.sort_index(inplace=True) 


    vocabulary_size = len(vocabulary)

    s = data['word'].value_counts()
    data['word'] = np.where(data['word'].isin(s.index[s < count_threshold]), '<UNK>', data['word'])

    print(f"Selected threshold: {count_threshold}\nTotal Size of Vocabulary: {vocabulary_size}\nNumber of occurences of <UNK>: {total_removed_words}")
    
    vocabulary['position_index'] = vocabulary.index

    return (data, vocabulary, vocabulary_size, total_words_before)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# class that is used to calculate the transitiona and emission probabilities
class HMM:
    def __init__(self, vocabulary = None, data = None) -> None:
        self.vocabulary = vocabulary
        self.data = data
        pass

    def generate_helpers(self):                                         # generate the list of tags, list of words, count for each tag
        self.data_dict = self.data.to_dict('records')
        self.unique_words = self.vocabulary['unique_words'].to_numpy()
        self.data['tag'].value_counts()
        self.tag_count = dict(zip(self.data['tag'].value_counts().index.tolist(), self.data['tag'].value_counts().tolist()))

        self.tag_list = self.data['tag'].unique().tolist()
        self.word_list = self.vocabulary['unique_words'].tolist()
        self.word_list.append("<START>")                                # add <START> to the vocabulary
    
    def train(self):

        self.generate_helpers()

        self.transition_count = {key:{key2:0 for key2 in self.tag_list} for key in self.tag_list}           # initialize dictionary to keep track of transition counts
        self.emission_count = {key:{key2:0 for key2 in self.word_list} for key in self.tag_list}            # initialize dictionary to keep track of emission counts

        count = 0
        temp_tag = '<START_TAG>'

        temp_row = {'position_index': '0', 'word': '<START>', 'tag': '<START_TAG>'}
        for row in self.data_dict:
            self.emission_count[str(row['tag'])][str(row['word'])] += 1
            self.transition_count[str(temp_tag)][str(row['tag'])] += 1

            temp_tag = row['tag']
            temp_row = row

            count += 1
            # if count % 100000 == 0:
            #     print(f"Completed {count} words.")
    
        self.emission_count['<START_TAG>']['<UNK>'] = self.tag_count['<START_TAG>']
        self.transition_count['<START_TAG>']['<START_TAG>'] -= 1

        self.calc_probability()
        self.convert_prob_dict_to_print()

        print(f"Number of non-zero Transition Parameters: {len(self.transition_prob_print)}\nNumber of non-zero Emission Parameters: {len(self.emission_prob_print)}")

    
    def calc_probability(self):
        # to find count for each tag based on emission and transition counts
        self.sum_transition_count = {key: sum(self.transition_count[str(key)].values()) for key in self.tag_list}
        self.sum_emission_count = {key: sum(self.emission_count[str(key)].values()) for key in self.tag_list}

        # to calculate emission probability accurately, there is a need to subtract the count since some words are set as unknown and therefore not considered
        transition_temp_dict = {}
        emission_temp_dict = {}
        for key in self.tag_list:
            transition_temp_dict[str(key)] = (self.tag_count[str(key)] - self.sum_transition_count[str(key)])
            emission_temp_dict[str(key)] = (self.tag_count[str(key)] - self.sum_emission_count[str(key)])

        self.tag_count_new = {key: val - emission_temp_dict[str(key)] for key,val in self.tag_count.items()}

        # calculate transition and emission probability
        self.transition_prob = {key: {key2: val/self.tag_count[str(key)] for key2, val  in self.transition_count[str(key)].items()} for key in self.tag_list}
        self.emission_prob = {key: {key2: val/self.tag_count_new[str(key)] for key2, val  in self.emission_count[str(key)].items()} for key in self.tag_list}
        
        # print("Done Training")

        # return (self.transition_prob, self.emission_prob)

    def convert_prob_dict_to_print(self):
        self.transition_prob_print = {}
        self.emission_prob_print = {}
        for tag1 in self.transition_prob.keys():
            for tag2 in self.transition_prob[tag1].keys():
                transmission_value = self.transition_prob[tag1][tag2]
                if (transmission_value != 0):
                    self.transition_prob_print[f"({tag1}, {tag2})"] = transmission_value
            for word in self.emission_prob[tag1].keys():
                emission_value = self.emission_prob[tag1][word]
                if (emission_value != 0):
                    self.emission_prob_print[f"({tag1}, {word})"] = emission_value
        
    def convert_print_to_prob_dict(self):
        self.transition_prob = {key:{key2:0 for key2 in self.tag_list} for key in self.tag_list}           # initialize dictionary to keep track of transition prob
        self.emission_prob = {key:{key2:0 for key2 in self.word_list} for key in self.tag_list}            # initialize dictionary to keep track of emission prob
        
        for word, prob in self.transition_prob_print.items():
            word = word[1:-1].split(" ")
            self.transition_prob[word[0][:-1]][word[1]] = prob

        for word, prob in self.emission_prob_print.items():
            word = word[1:-1].split(" ")
            self.emission_prob[word[0][0:-1]][word[1]] = prob

    def check_prob_sum(self):                                                           # to check if sum of each probability is 1
        sum_transition_prob = {key: sum(self.transition_prob[str(key)].values()) for key in self.tag_list}
        sum_emission_prob = {key: sum(self.emission_prob[str(key)].values()) for key in self.tag_list}

        return (sum_transition_prob, sum_emission_prob)

    def load_hmm(self, filepath):
        with open(filepath) as json_file:
            hmm = json.load(json_file)
        
        self.transition_prob_print = hmm['transition']
        self.emission_prob_print = hmm['emission']
        self.convert_print_to_prob_dict()
        self.tag_list = list(self.transition_prob.keys())

        return (self.transition_prob, self.emission_prob)

    def get_probability(self):
        return (self.transition_prob, self.emission_prob)

    def write_hmm_into_json(self, filepath):
        hmm = {'transition': self.transition_prob_print, 'emission': self.emission_prob_print}

        with open(filepath, "w") as json_file:
            json.dump(hmm, json_file, indent = 4)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# class to perfrom Greedy Decoding to the corpus given the transition and emission probablities
class GreedyDecoding:
    def __init__(self, data, tag_list, transition_prob, emission_prob, data_orig) -> None:
        self.data = data
        self.data_orig = data_orig
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.tag_list = tag_list
        pass

    def predict_sentence(self, sent_df):
        temp_tag = "<START_TAG>"
        sent = sent_df['word'].values.tolist()[1:]
        self.sent_predictions = []

        temp_prob = 1
        for word in sent:
            max_temp = -1
            for tag in self.tag_list:
                temp_prob = self.transition_prob[temp_tag][tag] * self.emission_prob[tag][word]
                if temp_prob > max_temp:
                    max_temp = temp_prob
                    corresponding_tag = tag
            
            temp_tag = corresponding_tag
            
            self.sent_predictions.append(corresponding_tag)

        return(self.sent_predictions, sent)
    
    def predict(self):
        self.predictions = []
        self.targets = []
        self.to_write = []
        
        count = 0
        for sent in self.data:
            sent_orig = self.data_orig[count]['word'].values.tolist()[1:]
            predictions, sent = self.predict_sentence(sent)
            pos = 1
            for pred, word in zip(predictions, sent_orig):
                self.to_write.append(f"{pos}\t{word}\t{pred}\n")
                pos += 1
            self.predictions.extend(predictions)
            self.to_write.append("\n")
            count += 1

        self.to_write = "".join(self.to_write[:-1])
        return self.predictions


    def calc_score(self, targets):
        count_of_matches = 0
        for pred, target in zip(self.predictions, targets):
            if pred == target:
                count_of_matches += 1
        self.accuracy = count_of_matches / len(self.predictions)
        return self.accuracy
    
    def get_targets(self):
        self.targets = []
        for sent_df in self.data:
            self.targets.extend(sent_df['tag'].values.tolist()[1:])
        return self.targets

    def write_prediction_into_file(self, filepath):
        with open(filepath, "w") as output_file:
            output_file.write(self.to_write)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# class to perfrom Viterbi Decoding to the corpus given the transition and emission probablities
class ViterbiDecoding:
    def __init__(self, data, tag_list, transition_prob, emission_prob, data_orig) -> None:
        self.data = data
        self.data_orig = data_orig
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob
        self.tag_list = tag_list[1:]
        self.map_tag_to_index()
        self.map_index_to_tag()
        pass

    def map_tag_to_index(self):
        self.tag_to_index = {}
        i = 0
        for tag in self.tag_list:
            self.tag_to_index[tag] = i
            i += 1

    def map_index_to_tag(self):
        self.index_to_tag = {v: k for k, v in self.tag_to_index.items()}

    def predict(self):
        self.predictions = []
        self.targets = []
        self.to_write = []

        count = 0
        for sent in self.data:
            sent_orig = self.data_orig[count]['word'].values.tolist()[1:]
            
            predictions, sent = self.predict_sentence(sent)
            pos = 1
            for pred, word in zip(predictions, sent_orig):
                self.to_write.append(f"{pos}\t{word}\t{pred}\n")
                pos += 1
            self.predictions.extend(predictions)
            
            self.to_write.append("\n")
            
            count += 1
            # if count % 500 == 0:
            #     print(f"Completed {count} sentences.")
        
        self.to_write = "".join(self.to_write[:-1])
        
        return self.predictions

    def calc_score(self, targets):
        count_of_matches = 0
        for pred, target in zip(self.predictions, targets):
            if pred == target:
                count_of_matches += 1

        self.accuracy = count_of_matches / len(self.predictions)
        return self.accuracy
    
    def get_targets(self):
        self.targets = []
        for sent_df in self.data:
            self.targets.extend(sent_df['tag'].values.tolist()[1:])
        return self.targets

    
    def predict_sentence(self, sent_df):
        self.sent = sent_df['word'].values.tolist()[1:]

        sentence_length = len(self.sent)
        no_of_tags = len(self.tag_list)

        self.OPT = np.zeros((no_of_tags, sentence_length))
        self.backtrack_matrix = np.zeros((no_of_tags, sentence_length))

        index = 0
        for tag in self.tag_list:
            self.OPT[self.tag_to_index[tag], index] = self.transition_prob['<START_TAG>'][tag] * self.emission_prob[tag][self.sent[index]]

        for j in range(1, sentence_length):
            for current_tag in self.tag_list:
                temp_prob = []
                for previous_tag in self.tag_list:
                    temp_prob.append(self.OPT[self.tag_to_index[previous_tag], j-1] * self.transition_prob[previous_tag][current_tag] * self.emission_prob[current_tag][self.sent[j]])
                
                max_tag_index = np.argmax(temp_prob)

                self.OPT[self.tag_to_index[current_tag], j] = temp_prob[max_tag_index]
                self.backtrack_matrix[self.tag_to_index[current_tag], j] = max_tag_index

        pred_tags = self.backtrack(self.OPT, self.backtrack_matrix)

        return (pred_tags, self.sent)

    def backtrack(self, OPT, bactrack_matrix):
        pred_tag = []
        sentence_length = len(self.sent)
        no_of_tags = len(self.tag_list)
        
        j = sentence_length - 1
        index = np.argmax(OPT[:,j])
        pointer = bactrack_matrix[index, j]
        pred_tag.append(self.index_to_tag[index])

        for j in range(sentence_length-2, -1, -1):
            pred_tag.append(self.index_to_tag[pointer])
            pointer = bactrack_matrix[int(pointer), j]

        pred_tag.reverse()

        return pred_tag
    
    def write_prediction_into_file(self, filepath):
        with open(filepath, "w") as output_file:
            output_file.write(self.to_write)


def main(direc = "/"):
    # print(f"{train_file_path} {dev_file_path} {train_file_path}")
    train_file_name = "/train"
    dev_file_name = "/dev"
    test_file_name = "/test"

    # Task 0: Reading and cleaning train data
    train_data, train_vocab = read_data(direc + train_file_name, get_vocab=True)

    # Task 1: Vocabulary Creation
    train_data, vocabulary, vocabulary_size, total_words_before = create_vocab(train_data, train_vocab, count_threshold=2)
    vocabulary.to_csv(direc + "/vocab.txt", index=None, header=None, sep='\t', columns=['unique_words', 'position_index', 'count'])     # writing vocabulary to vocab.txt

    # Task 2: Model Creation - Calculating Transition and Emission Probabilies
    hmm = HMM(vocabulary, train_data)
    hmm.train()
    hmm.write_hmm_into_json(direc + "/hmm.json")                           # writing the probabilites to hmm.json

    # Reading data for Task 3 and 4
    dev_data = read_data(direc + dev_file_name, get_vocab=False, separate_sentences=True, replace_unkown=True, vocabulary=hmm.vocabulary)
    dev_data_orig = read_data(direc + dev_file_name, get_vocab=False, separate_sentences=True, vocabulary=hmm.vocabulary)

    # fetching the probabilities to use for next two tasks
    transition_prob, emission_prob = hmm.get_probability()

    # Task 3: Greedy Decoding with HMM
    greedy_dev = GreedyDecoding(dev_data, hmm.tag_list, transition_prob, emission_prob, dev_data_orig)
    preds = greedy_dev.predict()
    acc = greedy_dev.calc_score(greedy_dev.get_targets())
    print(f"Greedy Decoding Accuracy on dev_data: {acc*100}")


    # Task 4: Viterbi Decoding with HMM
    viterbi_dev = ViterbiDecoding(dev_data, hmm.tag_list, transition_prob, emission_prob, dev_data_orig)
    preds = viterbi_dev.predict()
    acc = viterbi_dev.calc_score(viterbi_dev.get_targets())
    print(f"Viterbi Decoding Accuracy on dev_data: {acc*100}")

    # Reading data for Task 3 and 4
    test_data = read_data(direc + test_file_name, get_vocab=False, separate_sentences=True, replace_unkown=True, vocabulary=hmm.vocabulary)
    test_data_orig = read_data(direc + test_file_name, get_vocab=False, separate_sentences=True, vocabulary=hmm.vocabulary)

    # reading the probabilities from hmm.json to use for next two tasks
    transition_prob, emission_prob = hmm.load_hmm(direc + "/hmm.json")

    # Task 3: Greedy Decoding with HMM
    greedy_test = GreedyDecoding(test_data, hmm.tag_list, transition_prob, emission_prob, test_data_orig)
    preds = greedy_test.predict()
    greedy_test.write_prediction_into_file(direc + "/greedy.out")


    # Task 4: Viterbi Decoding with HMM
    viterbi_test = ViterbiDecoding(test_data, hmm.tag_list, transition_prob, emission_prob, test_data_orig)
    preds = viterbi_test.predict()
    viterbi_test.write_prediction_into_file(direc + "/viterbi.out")


    return


if __name__ == "__main__":
    n = len(sys.argv)
    if (n != 2):
        print(f"Insuffecient number of command line arguments.\nPlease provide ONE(1) argument. The argument must be the directory name where the train, dev and test files are present\n")
        sys.exit(-1)
    flag = True if n == 1 else False
    if flag:
        main()
    else:
        main(direc = sys.argv[1])
