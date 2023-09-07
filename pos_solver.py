###################################
# CS B551 Fall 2022, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
from collections import defaultdict


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:
    total_emission_count = {}
    total_transition_count = {}
    level_two_transition = {}
    level_two_emission = {}
    parts_of_speech = []
    viterbi_table = [{}]
    total_words_in_file = 0
    total_pos_count = defaultdict(int)

    emission_probability = {}
    transition_probability = {}
    level_two_transition_prob = {}
    level_two_emission_prob = {}
    parts_of_speech_list = {}

    verb_suffix = ["ify", "ize", "ate", "ish", "ise"]
    adj_suffix = ["tion", "sion", "ment", "ence", "ance", "like", "less", "able"]

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        words = list(sentence)
        tags = list(label)

        if model == "Simple":
            p = 0
            for i in range(len(words)):
                p += math.log10(self.populate_emission_probability(words[i], tags[i])) + math.log10(
                    self.parts_of_speech_list[tags[i]] / self.total_words_in_file)
            return p
        elif model == "HMM":
            transition_probability_s = 0
            emission_prob_at_instance = 0
            prob_s0 = math.log(
                self.parts_of_speech_list[tags[0]] / sum(self.parts_of_speech_list.values()),
                10)
            for i in range(len(words)):
                transition_probability_s += math.log(self.populate_emission_probability(words[i], tags[i]),
                                                         10)
                if i != 0:
                    emission_prob_at_instance += math.log(
                        self.populate_transition_probability(tags[i - 1], tags[i]), 10)
            return prob_s0 + transition_probability_s + emission_prob_at_instance
        elif model == "Complex":
            return self.calculate_mcmc_posterior(words, tags)
        else:
            print("Unknown algo!")

    # Do the training!
    #

    def train(self, data):
        previous_pos = None
        prev_previous_pos2 = None
        for (sentc, tags) in data:
            self.populate_initial_count_for_part_of_speech(tags[0])
            for i in range(len(sentc)):
                self.populate_emission_count(sentc[i], tags[i])

                if not previous_pos is None:
                    self.populate_transition_count(previous_pos, tags[i])
                if (prev_previous_pos2 and previous_pos) is not None:
                    self.calculate_level_two_emission(sentc[i], tags[i], previous_pos)
                    self.calculate_level_two_transition(tags[i], previous_pos, prev_previous_pos2)

                self.total_pos_count[tags[i]] += 1
                previous_pos = tags[i]
                try:
                    prev_previous_pos2 = tags[i - 1]
                except IndexError:
                    prev_previous_pos2 = None

        self.parts_of_speech = list(self.total_transition_count.keys())
        for i in range(len(self.total_pos_count)):
            self.total_words_in_file += self.total_pos_count[self.parts_of_speech[i]]

    def calculate_level_two_emission(self, word, tag, prev_tag):
        if word in self.level_two_emission:
            if prev_tag in self.level_two_emission[word]:
                if tag in self.level_two_emission[word][prev_tag]:
                    self.level_two_emission[word][prev_tag][tag] = self.level_two_emission[word][prev_tag][tag] + 1
                else:
                    self.level_two_emission[word][prev_tag][tag] = 1
            else:
                self.level_two_emission[word][prev_tag] = {tag: 1}
        else:
            self.level_two_emission[word] = {prev_tag: {tag: 1}}

    def calculate_level_two_transition(self, tag, prev_tag, prev_prev_tag):
        if tag in self.level_two_transition:
            if prev_prev_tag in self.level_two_transition[tag]:
                if prev_tag in self.level_two_transition[tag][prev_prev_tag]:
                    self.level_two_transition[tag][prev_prev_tag][prev_tag] = \
                        self.level_two_transition[tag][prev_prev_tag][prev_tag] + 1
                else:
                    self.level_two_transition[tag][prev_prev_tag][prev_tag] = 1
            else:
                self.level_two_transition[tag][prev_prev_tag] = {prev_tag: 1}
        else:
            self.level_two_transition[tag] = {prev_prev_tag: {prev_tag: 1}}

    def populate_emission_probability(self, word, part_of_speech):
        if word in self.emission_probability and part_of_speech in self.emission_probability[word]:
            return self.emission_probability[word][part_of_speech]

        if word in self.total_emission_count and part_of_speech in self.total_emission_count[
            word] and part_of_speech in self.total_transition_count:
            value = self.total_emission_count[word][part_of_speech] / sum(
                self.total_transition_count[part_of_speech].values())
            self.emission_probability[word] = {part_of_speech: value}
            return value
        return self.grammar_rules(word, part_of_speech)

    def grammar_rules(self, word, tag):
        p = 0.9
        try:
            if int(word):
                if tag == 'num':
                    return 1
        except ValueError:
            pass
        if word not in self.total_emission_count:
            if (list(word)[-3:] in self.verb_suffix or list(word)[-2:] == list("ed")) and tag == 'verb':
                return p

            if (list(word)[-4:] in self.adj_suffix or list(word)[-3:] == list("ful") or list(word)[-3:] == list("ous") or list(word)[-3:] == list("ish") or
                list(word)[-2:] == list("ic") or list(word)[-3:] == list("ive")) and tag == 'adj':
                return p

            if (list(word)[-2:] == list("ly")) and tag == 'adv':
                return p

            if (list(word)[-2:] == list("'s") or list(word)[-3:] == list("ist") or list(word)[-3:] == list("ion") or
                list(word)[-4:] == list("ment") or list(word)[-4:] == list("hood")) and tag == 'noun':
                return p

            if tag == 'noun':
                return 0.4
        return 0.0000001

    def populate_initial_probability(self, part_of_speech):
        if part_of_speech in self.parts_of_speech_list:
            return self.parts_of_speech_list[part_of_speech] / self.total_words_in_file
        return 0.00000001

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        tags_list = [''] * len(sentence)
        for j in range(len(sentence)):
            p = 0
            for i in range(len(self.parts_of_speech)):
                tag = self.parts_of_speech[i]
                p_value = self.populate_emission_probability(sentence[j], tag) * self.populate_initial_probability(tag)
                if p_value > p:
                    tags_list[j] = tag
                    p = p_value
        return tags_list

    def hmm_viterbi(self, sentence):
        state_sequence = {}
        for pos in self.parts_of_speech:
            self.viterbi_table[0][pos] = self.populate_initial_probability(pos) * self.populate_emission_probability(
                sentence[0], pos)
            state_sequence[pos] = [pos]

        for level1 in range(1, len(sentence)):
            self.viterbi_table.append({})
            current_path = {}

            for current_pos in self.parts_of_speech:
                max_value = 0
                for pre_pos in self.parts_of_speech:
                    value = self.viterbi_table[level1 - 1][pre_pos] * self.populate_transition_probability(
                        pre_pos, current_pos) * self.populate_emission_probability(sentence[level1], current_pos)
                    if value > max_value:
                        max_value = value
                        state = pre_pos
                self.viterbi_table[level1][current_pos] = max_value
                current_path[current_pos] = state_sequence[state] + [current_pos]

            state_sequence = current_path

        max_value = -math.inf
        last_level = len(sentence) - 1
        for pos in self.parts_of_speech:
            if self.viterbi_table[last_level][pos] >= max_value:
                max_value = self.viterbi_table[last_level][pos]
                best_state = pos
        state = best_state
        if len(state_sequence[state]) != len(sentence):
            print("HMM: Found mismatch for sentence:", sentence)
            print("Tags:", state_sequence[state])
            raise IndexError
        return state_sequence[state]

    def complex_mcmc(self, sentence):
        sample_count = 25
        burn_in_count = 10
        samples = []
        prev_sample = self.hmm_viterbi(sentence)
        for count in range(0, sample_count):
            curr_sample = self.copy_sample(prev_sample)
            for index in range(0, len(sentence)):
                log_prob_dict = {}
                prob_dict = {}
                for tag in self.parts_of_speech:
                    curr_sample[index] = tag
                    log_prob_dict[tag] = self.calculate_mcmc_posterior(sentence, curr_sample)
                    prob_dict[tag] = math.pow(10, log_prob_dict[tag])
                normalized_prob = {}
                for tag in self.parts_of_speech:
                    normalized_prob[tag] = prob_dict[tag]/sum(prob_dict.values())
                rand_val = random.random()
                cumulative_prob = 0
                for tag in self.parts_of_speech:
                    cumulative_prob = cumulative_prob + normalized_prob[tag]
                    if rand_val < cumulative_prob:
                        curr_sample[index] = tag
                        break
            if count > burn_in_count:
                samples.append(curr_sample)
            prev_sample = curr_sample
        tag_count = {}
        for index in range(0, len(sentence)):
            temp_dict = {}
            for tag in self.parts_of_speech:
                temp_dict[tag] = 0
            tag_count[index] = temp_dict
        for index in range(0, len(sentence)):
            for sample in samples:
                tag_count[index][sample[index]] = tag_count[index][sample[index]] + 1
        final_tag = []
        for index in range(0, len(sentence)):
            max_count = max(tag_count[index].values())
            for key in tag_count[index].keys():
                if tag_count[index][key] == max_count:
                    final_tag.append(key)
                    break
        return final_tag

    def copy_sample(self, sample):
        new_sample = ["" for _ in range(0, len(sample))]
        for i in range(0, len(sample)):
            new_sample[i] = sample[i]
        return new_sample

    def calculate_mcmc_posterior(self, sentence, tags):
        prob = math.log(self.parts_of_speech_list[tags[0]]/sum(self.parts_of_speech_list.values()), 10)
        prob = prob + math.log(self.populate_emission_probability(sentence[0], tags[0]), 10)
        if len(sentence) >= 2:
            transition_prob = math.log(self.populate_transition_probability(tags[0], tags[1]), 10)
            emission_prob = math.log(self.populate_level_two_emission_prob(sentence[1], tags[1], tags[0]), 10)
            prob = prob + transition_prob + emission_prob
        for i in range(2, len(sentence)):
            transition_prob_level = math.log(self.populate_level_two_transition_prob(tags[i], tags[i-1], tags[i-2]), 10)
            emission_prob = math.log(self.populate_level_two_emission_prob(sentence[i], tags[i], tags[i-1]), 10)
            prob = prob + transition_prob_level + emission_prob
        return prob

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

    def populate_initial_count_for_part_of_speech(self, part_of_speech):
        if part_of_speech in self.parts_of_speech_list:
            self.parts_of_speech_list[part_of_speech] = self.parts_of_speech_list[part_of_speech] + 1
        else:
            self.parts_of_speech_list[part_of_speech] = 1

    def get_viterbi_posterior(self, words, parts_of_speeches):
        transition_probability_s = 0
        emission_prob_at_instance = 0
        prob_s0 = math.log(self.parts_of_speech_list[parts_of_speeches[0]] / sum(self.parts_of_speech_list.values()),
                           10)
        for i in range(len(parts_of_speeches)):
            transition_probability_s += math.log(self.populate_emission_probability(words[i], parts_of_speeches[i]), 10)
            if i != 0:
                emission_prob_at_instance += math.log(
                    self.populate_transition_probability(parts_of_speeches[i - 1], parts_of_speeches[i]), 10)
        return prob_s0 + transition_probability_s + emission_prob_at_instance

    def populate_emission_count(self, word, part_of_speech):
        if word in self.total_emission_count:
            if part_of_speech in self.total_emission_count[word]:
                self.total_emission_count[word][part_of_speech] = self.total_emission_count[word][part_of_speech] + 1
            else:
                self.total_emission_count[word][part_of_speech] = 1
        else:
            self.total_emission_count[word] = {part_of_speech: 1}

    def populate_transition_count(self, pos_1, pos_2):
        if pos_1 in self.total_transition_count:
            if pos_2 in self.total_transition_count[pos_1]:
                self.total_transition_count[pos_1][pos_2] = self.total_transition_count[pos_1][pos_2] + 1
            else:
                self.total_transition_count[pos_1][pos_2] = 1
        else:
            self.total_transition_count[pos_1] = {pos_2: 1}

    def populate_transition_probability(self, part_of_speech1, part_of_speech2):
        if part_of_speech1 in self.transition_probability and part_of_speech2 in self.transition_probability[
            part_of_speech1]:
            return self.transition_probability[part_of_speech1][part_of_speech2]

        if part_of_speech1 in self.total_transition_count and part_of_speech2 in self.total_transition_count[
            part_of_speech1] and part_of_speech2 in self.total_transition_count:
            value = self.total_transition_count[part_of_speech1][part_of_speech2] / sum(
                self.total_transition_count[part_of_speech1].values())
            self.transition_probability[part_of_speech1] = {part_of_speech2: value}
            return value
        return 0.0000001

    def populate_level_two_transition_prob(self, tag, prev_tag, prev_prev_tag):
        if tag in self.level_two_transition_prob and prev_prev_tag in self.level_two_transition_prob[tag] and prev_tag in self.level_two_transition_prob[tag][prev_prev_tag]:
            return self.level_two_transition_prob[tag][prev_prev_tag][prev_tag]

        if tag in self.level_two_transition and prev_prev_tag in self.level_two_transition[tag] and prev_tag in self.level_two_transition[tag][prev_prev_tag]:
            prob = self.level_two_transition[tag][prev_prev_tag][prev_tag]/ sum(self.level_two_transition[tag][prev_prev_tag].values())
            self.level_two_transition_prob[tag] = {prev_prev_tag: {prev_tag: prob}}
            return self.level_two_transition_prob[tag][prev_prev_tag][prev_tag]

        return 0.0000001

    def populate_level_two_emission_prob(self, word, tag, prev_tag):
        if word in self.level_two_emission_prob and prev_tag in self.level_two_emission_prob[word] and tag in self.level_two_emission_prob[word][prev_tag]:
            return self.level_two_emission_prob[word][prev_tag][tag]

        if word in self.level_two_emission and prev_tag in self.level_two_emission[word] and tag in self.level_two_emission[word][prev_tag]:
            prob = self.level_two_emission[word][prev_tag][tag]/ sum(self.level_two_emission[word][prev_tag].values())
            self.level_two_emission_prob[word] = {prev_tag: {tag: prob}}
            return self.level_two_emission_prob[word][prev_tag][tag]

        return self.grammar_rules(word, tag)
