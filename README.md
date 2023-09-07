# POS_Tagging_bagofwords

### **Part 1: Part-of-speech tagging**

#### **Training**
During training, we read the training data and populate below data structures:
* **total_emission_count:** This is a dictionary of dictionary where, the keys of outer level dictionary corresponds to the words and the keys of the inner level dictionary corresponds to the part-of-speech. It contains the number of occurrences for each ordered pair (w, pos) in the training data. In other words, it stores the number of instances when word _w_ is tagged with part-of-speech _pos_ in the training data.
* **total_transition_count:** This is a dictionary of dictionary where, the keys of both outer level and inner level dictionary corresponds to the part-of-speech. It contains the number of occurrences for each ordered pair (pos1, pos2) in the training data. In other words, it stores the number of instances when part-of-speech _pos1_ is followed by the part-of-speech _pos2_ in the training data.
* **level_two_transition:** This is a three level dictionary where, the keys of all three levels corresponds to the part-of-speech. It contains the number of occurrences for each triplet (pos1, pos2, pos3) in the training data. In other words, it stores the number of instances when part-of-speech _pos1_ is followed by the part-of-speech _pos2_ followed by part-of-speech _pos3_ in the training data.
* **level_two_emission:** This is a three level dictionary where, the keys of the outer level dictionary corresponds to the words and the keys of other two levels corresponds to the part-of-speech. It contains the number of occurrences for each triplet (w, pos1, pos2) in the training data. In other words, it stores the number of instances when the word _w_ is tagged with part-of-speech _pos1_ and, _pos1_ is preceded by the part-of-speech _pos2_ in the training data.
* **parts_of_speech_list:** This is a dictionary where, the keys are the part-of-speech and value is the number of sentences in the training data where the part-of-speech occurred as the first tag of the sentence.
* **parts_of_speech:** It is a list containing all possible parts-of-speech.
* **total_pos_count:** This is a dictionary where, the keys are the part-of-speech and value is the number of occurrences of the part-of-speech in the training data.

#### **Simple Model:**
In this model, we return the maximized probability of the tag given the word. There is no relation between states. We only consider the emission probabilities. We use below expression for the posterior probability:

_si# = arg max-si P(Si = si|W)_

_P(si|wi) = {P(wi|Si) * P(Si)}/P(wi)_

The above expression contains below probabilities:
* **P(Si#):** The most probable tag si* for each Word Wi
* **P(Wi|Si):** This probability is calculated using the method _populate_emission_probability_ with the help of _total_emission_count_ data structure.

#### **Viterbi Model:**
For this model, we use below expression for the posterior probability:

_(s1#,.....sN#) =  arg max-s1,....,sN P(Si = si|W)_

_(p(Wi|Si) * p(Si))/p(Wi) = {p(w1|s1)p(w2|s2)..........p(wn|sN) Sum[(Si)/total words]}/sum(Wi/total words)_

The above expression contains below probabilities:
* **P(S1#):** This initial probability is calculated using _parts_of_speech_list_ data structure.
* **P(W1|S1):** initial probability of the word given start tag It is calculated using the method _populate_emission_probability_ with the help of _total_emission_count_ data structure.
* **P(Wi|Si):** This probability is calculated using the method _populate_emission_probability_ with the help of _total_emission_count_ data structure.
* **P(Wi|SiSi-1):** This probability is calculated using the method _populate_level_two_emission_prob_ with the help of _level_two_emission_ data structure.

#### **Complex Model:**
For this model, we use below expression for the posterior probability:
_P(S1-Sn|W1-Wn) = (P(S1)P(W1|S1)) * (P(W2|S1S2)P(S2|S1)) * (P(W3|S2S3)P(S3|S1S2))...(P(Wn|SnSn-1)P(Sn|Sn-1Sn-2))_

The above expression contains below probabilities:
* **P(S1):** This probability is calculated using _parts_of_speech_list_ data structure.
* **P(W1|S1):** This probability is calculated using the method _populate_emission_probability_ with the help of _total_emission_count_ data structure.
* **P(Wi|SiSi-1):** This probability is calculated using the method _populate_level_two_emission_prob_ with the help of _level_two_emission_ data structure.
* **P(Si|Si-1Si-2):** We calculate this probability using method _populate_level_two_transition_prob_ with the help of _level_two_transition_ data structure.

We used Gibbs Sampling to find the best tag sequence using complex model. The result of _hmm_viterbi_ is used as the initial sample. Using this initial sample, we generate 25 samples using below procedure:
* For each index _i_, we replace the tag at index _i_ with all possible tags one by one and calculate the posterior probability for the corresponding tag sequence. Note that, when changing the tag at index _i_. tags at all other indices are kept fixed.
* Among all tag sequences generated above, we choose one tag sequence randomly such that the probability of a tag sequence being selected is proportional to its corresponding posterior probability.
* The tag sequence selected above gives us the next sample in the Markov chain. 

Out of all samples generated using above procedure, we discard few initial samples. At the end we use these generated samples and for each index _i_, we count the number of times each tag appeared at index _i_. The tag having maximum number of occurrences at the index _i_ is assigned as the final tag of index _i_.

##### **Design Decisions**
* **Log probabilities:** The calculation of posterior probabilities requires multiplication of a significant number of probabilities which makes the final product to ba a very small number. To overcome this problem, we are using log probabilities in all three models.
* **Grammar rules:** During the execution of test data, if we encounter a word which is present in the training data, we use grammar rules to determine the _P(Wi|Si)_ and _P(Wi|SiSi-1)_ probabilities. Below are the specifications of the grammar rules:
  * Check the suffix of the word, if it falls under the category of a particular part-of-speech _pos_ and the tag is also _pos_ then, return 0.9 as probability.
  * If the word is an integer and the tag is _num_ then, return 1 as probability.
