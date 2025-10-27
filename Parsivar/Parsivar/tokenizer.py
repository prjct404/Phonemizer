import re
 


class Tokenizer():
    def __init__(self):
        pass

    def tokenize_words(self, doc_string):
        token_list = doc_string.strip().split()
        token_list = [x.strip("\u200c") for x in token_list if len(x.strip("\u200c")) != 0]
        return token_list

    def tokenize_sentences(self, doc_string):
        # finding the numbers
        pattern = r"[-+]?\d*\.\d+|\d+"
        print(doc_string)
        nums_list = re.findall(pattern, doc_string)
        doc_string = re.sub(pattern, 'floatingpointnumber', doc_string)

        pattern = r'([!\.\?؟]+)[\n]*'
        tmp = re.findall(pattern, doc_string)
        doc_string = re.sub(pattern, self.add_tab, doc_string)

        pattern = r':\n'
        tmp = re.findall(pattern, doc_string)
        doc_string = re.sub(pattern, self.add_tab, doc_string)

        pattern = r';[\n]*'
        tmp = re.findall(pattern, doc_string)
        doc_string = re.sub(pattern, self.add_tab, doc_string)

        pattern = r'؛[\n]*'
        tmp = re.findall(pattern, doc_string)
        doc_string = re.sub(pattern, self.add_tab, doc_string)

        pattern = r'^[\s\r\n]*$'
        doc_string = re.sub(pattern, '', doc_string)
        pattern = r'[\n\r]+'
        doc_string = re.sub(pattern, self.add_tab, doc_string)

        for number in nums_list:
            pattern = 'floatingpointnumber'
            doc_string = re.sub(pattern, number, doc_string, 1)

        doc_string = doc_string.split('\t\t')
        doc_string = [x for x in doc_string if len(x) > 0 and not x.isspace()]
        return doc_string

    def add_tab(self, mystring):
        mystring = mystring.group()  # this method return the string matched by re
        mystring = mystring.strip(' ')  # ommiting the whitespace around the pucntuation
        mystring = mystring.strip('\n')  # ommiting the newline around the pucntuation
        mystring = " " + mystring + "\t\t"  # adding a space after and before punctuation
        return mystring

    def tokenize_long_clauses(self, string_list, max_char_num):
        result = []
        pos_tagger = POSTagger('./g2p_resources/model/perpos.model')
        for string in string_list:
            tmp = string
            if len(tmp) > max_char_num:
                while (len(tmp) > max_char_num):
                    first_space_ind = tmp.find(' ', max_char_num)
                    second_space_ind = tmp.find(' ', 2*max_char_num)
                    pos = [pos_tagger.parse([word])[0] for word in tmp[first_space_ind:second_space_ind].split()]
                    try:
                        word_ind = [x[1] for x in pos[:20]].index('V')
                        split_index = sum(len(i[0]) for i in pos[:word_ind+1]) + word_ind + 1 + first_space_ind
                    except:
                        split_index = first_space_ind
                    result.append(tmp[:split_index])
                    tmp = tmp[split_index:]
                    if len(tmp) <= max_char_num:
                        result.append(tmp)
            else:
                result.append(tmp)

        return result
