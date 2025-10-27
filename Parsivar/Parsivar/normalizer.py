from re import sub
from unidecode import unidecode
import copy
import os
from .tokenizer import Tokenizer
from .data_helper import DataHelper
from .token_merger import ClassifierChunkParser
import pickle
import re


class Normalizer:

    def __init__(self,
                 half_space_char='\u200c',
                 statistical_space_correction=False,
                 date_normalizing_needed=True,
                 time_normalizing_needed=True,
                 pinglish_conversion_needed=False,
                 number2text_needed=True,
                 half_space_corrector=True,
                 train_file_path="resource/tokenizer/Bijan_khan_chunk.txt",
                 token_merger_path="resource/tokenizer/TokenMerger.pckl"):
        self.time_normalizing_needed = time_normalizing_needed
        self.dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"

        self.dic1_path = self.dir_path + 'resource/normalizer/Dic1_new.txt'
        self.dic2_path = self.dir_path + 'resource/normalizer/Dic2_new.txt'
        self.dic3_path = self.dir_path + 'resource/normalizer/Dic3_new.txt'
        self.dic1 = self.load_dictionary(self.dic1_path)
        self.dic2 = self.load_dictionary(self.dic2_path)
        self.dic3 = self.load_dictionary(self.dic3_path)

        self.tokenizer = Tokenizer()
        self.dictaition_normalizer = DictatioinNormalizer()
        self.statistical_space_correction = statistical_space_correction
        self.date_normalizing_needed = date_normalizing_needed
        self.pinglish_conversion_needed = pinglish_conversion_needed
        self.number2text_needed = number2text_needed
        self.half_space_corrector = half_space_corrector
        self.data_helper = DataHelper()
        self.token_merger = ClassifierChunkParser()

        if self.date_normalizing_needed or self.pinglish_conversion_needed:
            self.date_normalizer = DateNormalizer()
            self.pinglish_conversion = PinglishNormalizer()

        if self.time_normalizing_needed:
            self.time_normalizer = TimeNormalizer()

        if self.statistical_space_correction:
            self.token_merger_path = self.dir_path + token_merger_path
            self.train_file_path = train_file_path
            self.half_space_char = half_space_char

            if os.path.isfile(self.token_merger_path):
                self.token_merger_model = self.data_helper.load_var(self.token_merger_path)
            elif os.path.isfile(self.train_file_path):
                self.token_merger_model = self.token_merger.train_merger(self.train_file_path, test_split=0)
                self.data_helper.save_var(self.token_merger_path, self.token_merger_model)

    def load_dictionary(self, file_path):
        dictionary = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            g = f.readlines()
            for Wrds in g:
                wrd = Wrds.split(' ')
                dictionary[wrd[0].strip()] = sub('\n', '', wrd[1].strip())
        return dictionary

    def replace_abbrv(self, text):
        text = text.replace("( ص )", "( صلي‌الله‌عليه‌وآله )")
        text = text.replace("( ع )", "( عليه‌السلام )")
        text = text.replace("( س )", "( سلام‌الله )")
        text = text.replace("( ره )", "( رحمه‌الله‌عليه )")
        text = text.replace("( قده )", "( قدس‌سره )")
        text = text.replace("( رض )", "( رضي‌الله‌عنه )")
        text = text.replace(" ج ا ا ", " جمهوري اسلامي ايران ")
        text = text.replace(" ج اا ", " جمهوري اسلامي ايران ")
        text = text.replace(" ج‌اا ", " جمهوري اسلامي ايران ")
        text = text.replace(" ج ا ايران ", " جمهوري اسلامي ايران ")
        text = text.replace(" ج اايران ", " جمهوري اسلامي ايران ")
        text = text.replace(" ج‌اايران ", " جمهوري اسلامي ايران ")
        text = text.replace(" صص ", " صفحات ")
        text = text.replace(" ه ق ", " هجري قمري ")
        text = text.replace(" ه ش ", " هجري شمسي ")
        text = text.replace(" ق م ", " قبل از ميلاد ")
        text = text.replace(" صص ", " صفحات ")
        text = text.replace(" الخ ", " الي‌آخر ")
        return text

    def replace_special_symbols(self, doc_string):
        special_dict = {
            "﷼": " ریال ",
            "ﷴ": " محمد ",
            "ﷺ": " صلي‌الله‌عليه‌وآله‌وسلم ",
            "ﷲ": " الله ",
            "ﷻ": " جَل جلاله ",
            "ﷱ": " قلي ",
            "ﷳ": " صلي ",
            "ﷰ": " اكبر ",
            "ﷵ": " صلي‌الله‌عليه‌وآله‌وسلم ",
            "ﷶ": " رسول ",
            "ﷷ": " عليه‌السلام ",
            "ﷸ": " و سلم ",
            "﷽": " بسم‌الله‌الرحمن‌الرحيم ",
        }
        result = doc_string
        for i, j in special_dict.items():
            result = sub(i, j, result)
        return result

    def sub_alphabets(self, doc_string):
        # try:
        #     doc_string = doc_string.decode('utf-8')
        # except UnicodeEncodeError:
        #     pass
        a0 = "ء"
        b0 = "ئ"
        c0 = sub(a0, b0, doc_string)
        a1 = r"ٲ|ٱ|إ|ﺍ|أ"
        a11 = r"ﺁ|آ"
        b1 = r"ا"
        b11 = r"آ"
        c11 = sub(a11, b11, c0)
        c1 = sub(a1, b1, c11)
        a2 = r"ﺐ|ﺏ|ﺑ"
        b2 = r"ب"
        c2 = sub(a2, b2, c1)
        a3 = r"ﭖ|ﭗ|ﭙ|ﺒ|ﭘ"
        b3 = r"پ"
        c3 = sub(a3, b3, c2)
        a4 = r"ﭡ|ٺ|ٹ|ﭞ|ٿ|ټ|ﺕ|ﺗ|ﺖ|ﺘ"
        b4 = r"ت"
        c4 = sub(a4, b4, c3)
        a5 = r"ﺙ|ﺛ"
        b5 = r"ث"
        c5 = sub(a5, b5, c4)
        a6 = r"ﺝ|ڃ|ﺠ|ﺟ"
        b6 = r"ج"
        c6 = sub(a6, b6, c5)
        a7 = r"ڃ|ﭽ|ﭼ"
        b7 = r"چ"
        c7 = sub(a7, b7, c6)
        a8 = r"ﺢ|ﺤ|څ|ځ|ﺣ"
        b8 = r"ح"
        c8 = sub(a8, b8, c7)
        a9 = r"ﺥ|ﺦ|ﺨ|ﺧ"
        b9 = r"خ"
        c9 = sub(a9, b9, c8)
        a10 = r"ڏ|ډ|ﺪ|ﺩ"
        b10 = r"د"
        c10 = sub(a10, b10, c9)
        a11 = r"ﺫ|ﺬ|ﻧ"
        b11 = r"ذ"
        c11 = sub(a11, b11, c10)
        a12 = r"ڙ|ڗ|ڒ|ڑ|ڕ|ﺭ|ﺮ"
        b12 = r"ر"
        c12 = sub(a12, b12, c11)
        a13 = r"ﺰ|ﺯ"
        b13 = r"ز"
        c13 = sub(a13, b13, c12)
        a14 = r"ﮊ"
        b14 = r"ژ"
        c14 = sub(a14, b14, c13)
        a15 = r"ݭ|ݜ|ﺱ|ﺲ|ښ|ﺴ|ﺳ"
        b15 = r"س"
        c15 = sub(a15, b15, c14)
        a16 = r"ﺵ|ﺶ|ﺸ|ﺷ"
        b16 = r"ش"
        c16 = sub(a16, b16, c15)
        a17 = r"ﺺ|ﺼ|ﺻ"
        b17 = r"ص"
        c17 = sub(a17, b17, c16)
        a18 = r"ﺽ|ﺾ|ﺿ|ﻀ"
        b18 = r"ض"
        c18 = sub(a18, b18, c17)
        a19 = r"ﻁ|ﻂ|ﻃ|ﻄ"
        b19 = r"ط"
        c19 = sub(a19, b19, c18)
        a20 = r"ﻆ|ﻇ|ﻈ"
        b20 = r"ظ"
        c20 = sub(a20, b20, c19)
        a21 = r"ڠ|ﻉ|ﻊ|ﻋ"
        b21 = r"ع"
        c21 = sub(a21, b21, c20)
        a22 = r"ﻎ|ۼ|ﻍ|ﻐ|ﻏ"
        b22 = r"غ"
        c22 = sub(a22, b22, c21)
        a23 = r"ﻒ|ﻑ|ﻔ|ﻓ"
        b23 = r"ف"
        c23 = sub(a23, b23, c22)
        a24 = r"ﻕ|ڤ|ﻖ|ﻗ"
        b24 = r"ق"
        c24 = sub(a24, b24, c23)
        a25 = r"ڭ|ﻚ|ﮎ|ﻜ|ﮏ|ګ|ﻛ|ﮑ|ﮐ|ڪ|ک"
        b25 = r"ك"
        c25 = sub(a25, b25, c24)
        a26 = r"ﮚ|ﮒ|ﮓ|ﮕ|ﮔ"
        b26 = r"گ"
        c26 = sub(a26, b26, c25)
        a27 = r"ﻝ|ﻞ|ﻠ|ڵ"
        b27 = r"ل"
        c27 = sub(a27, b27, c26)
        a28 = r"ﻡ|ﻤ|ﻢ|ﻣ"
        b28 = r"م"
        c28 = sub(a28, b28, c27)
        a29 = r"ڼ|ﻦ|ﻥ|ﻨ"
        b29 = r"ن"
        c29 = sub(a29, b29, c28)
        a30 = r"ވ|ﯙ|ۈ|ۋ|ﺆ|ۊ|ۇ|ۏ|ۅ|ۉ|ﻭ|ﻮ|ؤ"
        b30 = r"و"
        c30 = sub(a30, b30, c29)
        a31 = r"ﺔ|ﻬ|ھ|ﻩ|ﻫ|ﻪ|ۀ|ە|ة|ہ|\u06C1"
        b31 = r"ه"
        c31 = sub(a31, b31, c30)
        a32 = r"ﭛ|ﻯ|ۍ|ﻰ|ﻱ|ﻲ|ں|ﻳ|ﻴ|ﯼ|ې|ﯽ|ﯾ|ﯿ|ێ|ے|ى|ی"
        b32 = r"ي"
        c32 = sub(a32, b32, c31)
        a33 = r'¬'
        b33 = r'‌'
        c33 = sub(a33, b33, c32)
        pa0 = r'•|·|●|·|・|∙|｡|ⴰ'
        pb0 = r'.'
        pc0 = sub(pa0, pb0, c33)
        pa1 = r',|٬|‚|，'
        pb1 = r'،'
        pc1 = sub(pa1, pb1, pc0)
        pa2 = r'ʕ'
        pb2 = r'؟'
        pc2 = sub(pa2, pb2, pc1)
        pa3 = r'٪'
        pb3 = r'%'
        pc3 = sub(pa3, pb3, pc2)
        na0 = r'۰|٠'
        nb0 = r'0'
        nc0 = sub(na0, nb0, pc3)
        na1 = r'۱|١'
        nb1 = r'1'
        nc1 = sub(na1, nb1, nc0)
        na2 = r'۲|٢'
        nb2 = r'2'
        nc2 = sub(na2, nb2, nc1)
        na3 = r'۳|٣'
        nb3 = r'3'
        nc3 = sub(na3, nb3, nc2)
        na4 = r'۴|٤'
        nb4 = r'4'
        nc4 = sub(na4, nb4, nc3)
        na5 = r'۵'
        nb5 = r'5'
        nc5 = sub(na5, nb5, nc4)
        na6 = r'۶|٦'
        nb6 = r'6'
        nc6 = sub(na6, nb6, nc5)
        na7 = r'۷|٧'
        nb7 = r'7'
        nc7 = sub(na7, nb7, nc6)
        na8 = r'۸|٨'
        nb8 = r'8'
        nc8 = sub(na8, nb8, nc7)
        na9 = r'۹|٩'
        nb9 = r'9'
        nc9 = sub(na9, nb9, nc8)
        ea1 = r'ـ|ِ|ُ|َ'
        eb1 = r''
        #ec1 = sub(ea1, eb1, nc9)
        ec1 = nc9
        sa1 = r'( )+'
        sb1 = r' '
        sc1 = sub(sa1, sb1, ec1)
        sa2 = r'(\n)+'
        sb2 = r'\n'
        sc2 = sub(sa2, sb2, sc1)
        espa0 = u'\u200e|\u200f| '
        espb0 = ' '
        espc0 = sub(espa0, espb0, sc2)
        return espc0

    def space_correction(self, doc_string):
        a00 = r'^(بی|می|نمی)( )'
        b00 = r'\1‌'
        c00 = sub(a00, b00, doc_string)
        a0 = r'( )(می|نمی|بی)( )'
        b0 = r'\1\2‌'
        c0 = sub(a0, b0, c00)
        a1 = r'( )(هایی|ها|های|ایی|هایم|هایت|هایش|هایمان|هایتان|هایشان|ات|ان|ین' \
             r'|انی|بان|ام|ای|یم|ید|اید|اند|بودم|بودی|بود|بودیم|بودید|بودند|ست)( )'
        b1 = r'‌\2\3'
        c1 = sub(a1, b1, c0)
        a2 = r'( )(شده|نشده)( )'
        b2 = r'‌\2‌'
        c2 = sub(a2, b2, c1)
        a3 = r'( )(طلبان|طلب|گرایی|گرایان|شناس|شناسی|گذاری|گذار|گذاران|شناسان|گیری|پذیری|بندی|آوری|سازی|' \
             r'بندی|کننده|کنندگان|گیری|پرداز|پردازی|پردازان|آمیز|سنجی|ریزی|داری|دهنده|آمیز|پذیری' \
             r'|پذیر|پذیران|گر|ریز|ریزی|رسانی|یاب|یابی|گانه|گانه‌ای|انگاری|گا|بند|رسانی|دهندگان|دار)( )'
        b3 = r'‌\2\3'
        c3 = sub(a3, b3, c2)
        return c3

    def space_correction_plus1(self, doc_string):
        out_sentences = ''
        for wrd in doc_string.split(' '):
            try:
                out_sentences = out_sentences + ' ' + self.dic1[wrd]
            except KeyError:
                out_sentences = out_sentences + ' ' + wrd
        return out_sentences

    def space_correction_plus2(self, doc_string):
        out_sentences = ''
        wrds = doc_string.split(' ')
        word_len = wrds.__len__()
        if word_len < 2:
            return doc_string
        cnt = 1
        for i in range(0, word_len - 1):
            w = wrds[i] + wrds[i + 1]
            try:
                out_sentences = out_sentences + ' ' + self.dic2[w]
                cnt = 0
            except KeyError:
                if cnt == 1:
                    out_sentences = out_sentences + ' ' + wrds[i]
                cnt = 1
        if cnt == 1:
            out_sentences = out_sentences + ' ' + wrds[i + 1]
        return out_sentences

    def space_correction_plus3(self, doc_string):
        # Dict = {'گفتوگو': 'گفت‌وگو'}
        out_sentences = ''
        wrds = doc_string.split(' ')
        word_len = wrds.__len__()
        if word_len < 3:
            return doc_string
        cnt = 1
        cnt2 = 0
        for i in range(0, word_len - 2):
            w = wrds[i] + wrds[i + 1] + wrds[i + 2]
            try:
                out_sentences = out_sentences + ' ' + self.dic3[w]
                cnt = 0
                cnt2 = 2
            except KeyError:
                if cnt == 1 and cnt2 == 0:
                    out_sentences = out_sentences + ' ' + wrds[i]
                else:
                    cnt2 -= 1
                cnt = 1
        if cnt == 1 and cnt2 == 0:
            out_sentences = out_sentences + ' ' + wrds[i + 1] + ' ' + wrds[i + 2]
        elif cnt == 1 and cnt2 == 1:
            out_sentences = out_sentences + ' ' + wrds[i + 2]
        return out_sentences

    def replace_puncs(self, doc_string):
        repeat_pattern = \
            r"((?<!\.)\.(?!\.)|\.{3}\.*|_|،|,|\(|\)|:|\?|!|<|>|\-|;|\[|\]|\{|\}|»|«|\^|'|\\|¡|~|©|،|؟|؛|\")\1+"
        doc_strig = re.sub(repeat_pattern, r"\1", doc_string)
        pattern = r'(?<!\.)\.(?!\.)|\.{3}\.*|_|،|,|\(|\)|:|\?|!|<|>|\-|;|\[|\]|\{|\}|»|«|\^|\'|\\|¡|~|©|،|؟|؛|\"'
        clauses = [i.strip() for i in list(filter(None, re.split(pattern, doc_string)))]
        result = ' | '.join(clauses)
        return result

    def split_digit_from_alphabet(self, doc_string):
        doc_string = re.sub(r'(\d)([^\d\s])', r'\1 \2', doc_string)
        doc_string = re.sub(r'(\D)([^\D\s])', r'\1 \2', doc_string)
        return doc_string

    def normalize(self, doc_string, new_line_elimination=False):
        normalized_string = doc_string
        normalized_string = self.dictaition_normalizer.remove_extra_space_zwnj(normalized_string)
        normalized_string = self.sub_alphabets(normalized_string)
        normalized_string = self.replace_abbrv(normalized_string)
        normalized_string = self.replace_special_symbols(normalized_string)
        normalized_string = self.data_helper.clean_text(normalized_string, new_line_elimination).strip()
        normalized_string = self.split_digit_from_alphabet(normalized_string)
        normalized_string = self.split_phoneme_in_persian_and_eng_from_rest(normalized_string)
        if self.date_normalizing_needed:
            normalized_string = self.date_normalizer.normalize_dates(normalized_string)
        if self.time_normalizing_needed:
            normalized_string = self.time_normalizer.normalize_time(normalized_string)

        if self.number2text_needed:
            normalized_string = sub(r'[\u06F0-\u06F90-9]+', lambda x: unidecode(x.group(0)), normalized_string)
            n = NumberNormalizer()
            normalized_string = sub(r'[0-9]+', lambda x: n.convert(x.group(0)), normalized_string)
        if self.statistical_space_correction:
            token_list = normalized_string.strip().split()
            token_list = [x.strip("\u200c") for x in token_list if len(x.strip("\u200c")) != 0]
            token_list = self.token_merger.merg_tokens(token_list, self.token_merger_model, self.half_space_char)
            normalized_string = " ".join(x for x in token_list)
            normalized_string = self.data_helper.clean_text(normalized_string, new_line_elimination)
        else:
            normalized_string = self.space_correction(self.space_correction_plus1(self.space_correction_plus2(
                self.space_correction_plus3(normalized_string)))).strip()
        if self.pinglish_conversion_needed:
            normalized_string = self.pinglish_conversion.pingilish2persian(
                self.tokenizer.tokenize_words(normalized_string))
        normalized_string = self.replace_puncs(normalized_string)
        if self.half_space_corrector:
            normalized_string = self.dictaition_normalizer.remove_extra_space_zwnj(normalized_string)
            normalized_string = self.dictaition_normalizer.join_words_without_rules(normalized_string)
            normalized_string = self.dictaition_normalizer.correct_compound(normalized_string, 5)
            normalized_string = self.dictaition_normalizer.half_space_corrector(normalized_string)

        return normalized_string

    def split_phoneme_in_persian_and_eng_from_rest(self, doc_string):
        doc_string = re.sub(r'([\u0622-\u06f0])([^\u0622-\u06f0])', r'\1 \2', doc_string)
        doc_string = re.sub(r'([a-z|A-Z])([^a-z|^A-Z])', r'\1 \2', doc_string)
        return doc_string


class DateNormalizer:
    def __init__(self):
        self.nn = NumberNormalizer()
        self.persian_month_dict = {'فروردين': 1, 'ارديبهشت': 2, 'خرداد': 3, 'تير': 4, 'مرداد': 5,
                                   'شهريور': 6, 'مهر': 7, 'آبان': 8, 'آذر': 9, 'دي': 10, 'بهمن': 11, 'اسفند': 12}
        self.christian_month_dict = {'ژانویه': 1, 'فوریه': 2, 'مارس': 3, 'آپریل': 4, 'می': 5, 'ژوئن': 6, 'جولای': 7,
                                     'آگوست': 8, 'سپتامبر': 9, 'اکتبر': 10, 'نوامبر': 11, 'دسامبر': 12}

        self.num_dict = {'چهار': 4, 'سه': 3, 'دو': 2, 'يك': 1, 'يازده': 11, 'سيزده': 13, 'چهارده': 14, 'دوازده': 12,
                         'پانزده': 15, 'شانزده': 16, 'چهارم': 4, 'سوم': 3, 'دوم': 2, 'يكم': 1, 'اول': 1, 'يازدهم': 11,
                         'سيزدهم': 13, 'چهاردهم': 14, 'دوازدهم': 12, 'پانزدهم': 15, 'شانزدهم': 16, 'هفدهم': 17,
                         'هجدهم': 18, 'نوزدهم': 19, 'بيستم': 20, 'چهلم': 40, 'پنجاهم': 50, 'شصتم': 60, 'هفتادم': 70,
                         'نودم': 90, 'سيصدم': 300, 'چهارصدم': 400, 'پانصدم': 500, 'ششصدم': 600, 'هفتصدم': 700,
                         'هشتصدم': 800, 'نهصدم': 900, 'هشتادم': 80, 'هزار': 1000, 'ميليون': 1000000, 'دويست': 200,
                         'ده': 10, 'نه': 9, 'هشت': 8, 'هفت': 7, 'شش': 6, 'پنج': 5, 'هفده': 17, 'هجده': 18, 'نوزده': 19,
                         'بيست': 20, 'سي': 30, 'چهل': 40, 'پنجاه': 50, 'شصت': 60, 'هفتاد': 70, 'نود': 90, 'سيصد': 300,
                         'چهارصد': 400, 'پانصد': 500, 'ششصد': 600, 'هفتصد': 700, 'هشتصد': 800, 'نهصد': 900,
                         'هشتاد': 80, ' ': 0, 'ميليارد': 1000000000, 'صدم': 100, 'هزارم': 1000, 'دويستم': 200,
                         'دهم': 10, 'نهم': 9, 'هشتم': 8, 'هفتم': 7, 'ششم': 6, 'پنجم': 5}

    def find_date_part(self, doc_string):
        persian_date_regex = re.compile(r'\b(1[0-4]\d\d|[3-9][2-9])([\s]*[/.\-][\s]*)([1-9]|0[1-9]|1[0-2])'
                                        r'([\s]*[/.\-][\s]*)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])\b')
        persian_date_regex_rev = re.compile(
            r'\b([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])([\s]*[/.\-][\s]*)([1-9]|0[1-9]|1[0-2])'
            r'([\s]*[/.-][\s]*)(1[0-4][\d][\d]|[3-9][2-9])\b')
        persian_date_md_regex = re.compile(r'(?<![/.\-])([0[1-9]|[1-9]|1[0-2])([\s]*[/.\-][\s]*)'
                                           r'([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])[\s]*(?![/.\-])')
        christian_date_regex = re.compile(r'\b([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])([\s]*[/.\-][\s]*)'
                                          r'([1-9]|0[1-9]|1[0-2])([\s]*[/.\-][\s]*)(1[5-9][0-9][0-9]|2[0][0-9][0-9])|'
                                          r'([1-9]|0[1-9]|1[0-2])([\s]*[/.\-][\s]*)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])'
                                          r'([\s]*[/.\-][\s]*)(1[5-9][0-9][0-9]|2[0][0-9][0-9])|'
                                          r'(1[5-9][0-9][0-9]|2[0][0-9][0-9])([\s]*[/.\-][\s]*)([1-9]|0[1-9]|1[0-2])'
                                          r'([\s]*[/.\-][\s]*)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])\b')
        keywords_date = ['مورخ', 'مورخه', 'تاريخ', 'شمسي', 'ميلادي', 'قمري', 'هجري']
        persian_result = []
        christian_result = []
        for match in persian_date_regex.finditer(doc_string):
            persian_result.append([*match.span(), int(match[5]), int(match[3]), int(match[1])])
        for match in persian_date_regex_rev.finditer(doc_string):
            persian_result.append([*match.span(), int(match[1]), int(match[3]), int(match[5])])

        for match in christian_date_regex.finditer(doc_string):
            a = False
            for s in persian_result:
                if range(max(s[0], match.start()), min(s[1], match.end())):
                    a = True
            if not a:
                if match[11] is not None:
                    christian_result.append([*match.span(), int(match[15]), int(match[13]), int(match[11])])
                elif match[6] is not None:
                    christian_result.append([*match.span(), int(match[10]), int(match[8]), int(match[6])])
                elif match[1] is not None:
                    christian_result.append([*match.span(), int(match[5]), int(match[3]), int(match[1])])

        for match in persian_date_md_regex.finditer(doc_string):
            if ngram_lookup(doc_string, match.start(), match.end(), keywords_date, 3) and \
                    not ngram_lookup(doc_string, match.start(), match.end(), keywords_date, 3):
                a = False
                for s in persian_result + christian_result:
                    if range(max(s[0], match.start()), min(s[1], match.end())):
                        a = True
                if not a:
                    persian_result.append([*match.span(), int(match[3]), int(match[1])])

        return [sorted(persian_result), sorted(christian_result)]

    def date_to_text_persian(self, doc_string, finded_dates, christian_dates):
        for i in range(len(finded_dates)):
            finded_date = finded_dates[i]
            str_date = ''
            str_date += self.nn.convert_ordinary(finded_date[2]) + ' '
            str_date += list(self.persian_month_dict.keys())[int(finded_date[3]) - 1] + ' '
            if len(finded_date) == 5:
                str_date += self.nn.convert(finded_date[4]) + ' '
            if str_date and str_date[-1] != ' ':
                str_date += ' '
            start_date_index = finded_date[0]
            end_date_index = finded_date[1]
            doc_string = doc_string[:start_date_index] + str_date + doc_string[end_date_index:]
            for j in range(i + 1, len(finded_dates)):
                finded_dates[j][0] += len(str_date) - (end_date_index - start_date_index)
                finded_dates[j][1] += len(str_date) - (end_date_index - start_date_index)
            for j in range(i + 1, len(christian_dates)):
                christian_dates[j][0] += len(str_date) - (end_date_index - start_date_index)
                christian_dates[j][1] += len(str_date) - (end_date_index - start_date_index)
        return doc_string, christian_dates

    def date_to_text_christian(self, doc_string, finded_dates):
        for i in range(len(finded_dates)):
            finded_date = finded_dates[i]
            str_date = ''
            str_date += self.nn.convert_ordinary(finded_date[2]) + ' '
            str_date += list(self.christian_month_dict.keys())[int(finded_date[3]) - 1] + ' '
            if len(finded_date) == 5:
                str_date += self.nn.convert(finded_date[4]) + ' '
            if str_date and str_date[-1] != ' ':
                str_date += ' '
            start_date_index = finded_date[0]
            end_date_index = finded_date[1]
            doc_string = doc_string[:start_date_index] + str_date + doc_string[end_date_index:]
            for j in range(i + 1, len(finded_dates)):
                finded_dates[j][0] += len(str_date) - (end_date_index - start_date_index)
                finded_dates[j][1] += len(str_date) - (end_date_index - start_date_index)
        return doc_string

    def normalize_dates(self, doc_string):
        finded_dates = self.find_date_part(doc_string)
        doc_string, finded_dates[1] = self.date_to_text_persian(doc_string, finded_dates[0], finded_dates[1])
        doc_string = self.date_to_text_christian(doc_string, finded_dates[1])
        return doc_string

    def list2num(self, numerical_section_list):
        value = 1
        for index, el in enumerate(numerical_section_list):
            if self.is_number(el):
                value *= self.num_dict[el]
            else:
                value *= float(el)
        return value

    def convert2num(self, numerical_section_list):
        value = 0
        tmp_section_list = []
        for index, el in enumerate(numerical_section_list):
            if self.is_number(el) or (el.replace('.', '', 1).isdigit()):
                tmp_section_list.append(el)
            elif el == "و":
                value += self.list2num(tmp_section_list)
                tmp_section_list[:] = []
        if len(tmp_section_list) > 0:
            value += self.list2num(tmp_section_list)
            tmp_section_list[:] = []
        if value - int(value) == 0:
            return int(value)
        else:
            return value

    def is_number(self, word):
        return word in self.num_dict

    def find_number_location(self, token_list):
        start_index = 0
        number_section = []
        for i, el in enumerate(token_list):
            if self.is_number(el) or (el.replace('.', '', 1).isdigit()):
                start_index = i
                number_section.append(start_index)
                break

        i = start_index + 1
        while i < len(token_list):
            if token_list[i] == "و" and (i + 1) < len(token_list):
                if self.is_number(token_list[i + 1]) or (token_list[i + 1].replace('.', '', 1).isdigit()):
                    number_section.append(i)
                    number_section.append(i + 1)
                    i += 2
                else:
                    break
            elif self.is_number(token_list[i]) or (token_list[i].replace('.', '', 1).isdigit()):
                number_section.append(i)
                i += 1
            else:
                break
        return number_section

    def normalize_numbers(self, token_list, converted=""):
        for i, el in enumerate(token_list):
            if el.endswith("ین") and self.is_number(el[:-2]):
                token_list[i] = el[:-2]
        finded = self.find_number_location(token_list)
        if len(finded) == 0:
            rest_of_string = " ".join(t for t in token_list)
            return converted + " " + rest_of_string
        else:
            numerical_subsection = [token_list[x] for x in finded]
            numerical_subsection = self.convert2num(numerical_subsection)

            converted = converted + " " + " ".join(x for x in token_list[:finded[0]]) + " " + str(numerical_subsection)

            new_index = finded[-1] + 1
            return self.normalize_numbers(token_list[new_index:], converted)


class NumberNormalizer:
    def __init__(self):
        self.faBaseNum = {1: 'يك', 2: 'دو', 3: 'سه', 4: 'چهار', 5: 'پنج', 6: 'شِش', 7: 'هفت', 8: 'هشت', 9: 'نُه',
                          10: 'دَه', 11: 'يازده', 12: 'دوازده', 13: 'سيزده', 14: 'چهارده', 15: 'پانزده', 16: 'شانزده',
                          17: 'هفده', 18: 'هجده', 19: 'نوزده', 20: 'بيست', 30: 'سي', 40: 'چهل', 50: 'پنجاه', 60: 'شصت',
                          70: 'هفتاد', 80: 'هشتاد', 90: 'نود', 100: 'صد', 200: 'دويست', 300: 'سيصد', 500: 'پانصد'}

        self.faBaseNumKeys = self.faBaseNum.keys()
        self.faBigNum = ["يك", "هزار", "ميليون", "ميليارد"]
        self.faBigNumSize = len(self.faBigNum)

    def split3(self, st):
        parts = []
        n = len(st)
        d, m = divmod(n, 3)
        for i in range(d):
            parts.append(int(st[n - 3 * i - 3:n - 3 * i]))
        if m > 0:
            parts.append(int(st[:m]))
        return parts

    def convert(self, st):
        st = str(st)
        if len(st) > 3:
            parts = self.split3(st)
            k = len(parts)
            wparts = []
            for i in range(k):
                p = parts[i]
                if p == 0:
                    continue
                if i == 0:
                    wpart = self.convert(p)
                else:
                    if i < self.faBigNumSize:
                        fa_order = self.faBigNum[i]
                    else:
                        fa_order = ''
                        (d, m) = divmod(i, 3)
                        t9 = self.faBigNum[3]
                        for j in range(d):
                            if j > 0:
                                fa_order += "‌"
                            fa_order += t9
                        if m != 0:
                            if fa_order != '':
                                fa_order = "‌" + fa_order
                            fa_order = self.faBigNum[m] + fa_order
                    wpart = fa_order if i == 1 and p == 1 else self.convert(p) + " " + fa_order
                wparts.append(wpart)
            return "  وَ ".join(reversed(wparts))
        # now assume that n <= 999
        n = int(st)
        if n in self.faBaseNumKeys:
            return self.faBaseNum[n]
        y = n % 10
        d = int((n % 100) / 10)
        s = int(n / 100)
        # print s, d, y
        dy = 10 * d + y
        fa = ''
        if s != 0:
            if s * 100 in self.faBaseNumKeys:
                fa += self.faBaseNum[s * 100]
            else:
                fa += (self.faBaseNum[s] + self.faBaseNum[100])
            if d != 0 or y != 0:
                fa += "  وَ "
        if d != 0:
            if dy in self.faBaseNumKeys:
                fa += self.faBaseNum[dy]
                return fa
            fa += self.faBaseNum[d * 10]
            if y != 0:
                fa += "  وَ "
        if y != 0:
            fa += self.faBaseNum[y]
        return fa

    def convert_ordinary(self, arg):
        if isinstance(arg, int):
            num = arg
            st = str(arg)
        elif isinstance(arg, str):
            num = int(arg)
            st = arg
        else:
            raise TypeError('bad type "%s"' % type(arg))
        if num == 1:
            return 'اول'  # OR 'یکم'fixme
        elif num == 10:
            return 'دهم'
        norm_fa = self.convert(st)
        if len(norm_fa) == 0:
            return ''
        if norm_fa.endswith(u'ی'):
            norm_fa += u'‌ام'
        elif norm_fa.endswith(u'سه'):
            norm_fa = norm_fa[:-1] + u'وم'
        else:
            norm_fa += u'م'
        return norm_fa


class PinglishNormalizer:
    def __init__(self):
        self.data_helper = DataHelper()
        self.file_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

        self.en_dict_filename = self.file_dir + "resource/tokenizer/enDict"
        self.en_dict = self.data_helper.load_var(self.en_dict_filename)

        self.fa_dict_filename = self.file_dir + "resource/tokenizer/faDict"
        self.fa_dict = self.data_helper.load_var(self.fa_dict_filename)

    def pingilish2persian(self, pinglish_words_list):

        for i, word in enumerate(pinglish_words_list):
            if word in self.en_dict:
                pinglish_words_list[i] = self.en_dict[word]  # .decode("utf-8")
                # inp = inp.replace(word, enDict[word], 1)
            else:
                ch = self.characterize(word)
                pr = self.map_char(ch)
                amir = self.make_word(pr)
                for wd in amir:
                    am = self.escalation(wd)
                    asd = ''.join(am)
                    if asd in self.fa_dict:
                        pinglish_words_list[i] = asd  # .decode("utf-8")
                        # inp = inp.replace(word, asd, 1)
        inp = " ".join(x for x in pinglish_words_list)
        return inp

    def characterize(self, word):
        list_of_char = []
        i = 0
        while i < len(word):
            char = word[i]
            sw_out = self.switcher(char)
            if sw_out is None:
                esp_out = None
                if i < len(word) - 1:
                    esp_out = self.esp_check(word[i], word[i + 1])
                if esp_out is None:
                    list_of_char.append(word[i])
                else:
                    list_of_char.append(esp_out)
                    i += 1
            else:
                list_of_char.append(sw_out)
            i += 1
        return list_of_char

    def switcher(self, ch):
        switcher = {
            "c": None,
            "k": None,
            "z": None,
            "s": None,
            "g": None,
            "a": None,
            "u": None,
            "e": None,
            "o": None
        }
        return switcher.get(ch, ch)

    def esp_check(self, char1, char2):
        st = char1 + char2
        if st == "ch":
            return "ch"
        elif st == "kh":
            return "kh"
        elif st == "zh":
            return "zh"
        elif st == "sh":
            return "sh"
        elif st == "gh":
            return "gh"
        elif st == "aa":
            return "aa"
        elif st == "ee":
            return "ee"
        elif st == "oo":
            return "oo"
        elif st == "ou":
            return "ou"
        else:
            return None

    def map_char(self, word):
        listm = []
        sw_out = self.map_switcher(word[0])
        i = 0
        if sw_out is None:
            listm.append(["ا"])
            i += 1
        if word[0] == "oo":
            listm.append(["او"])
            i += 1
        while i < len(word):
            listm.append(self.char_switcher(word[i]))
            i += 1
        if word[len(word) - 1] == "e":
            listm.append(["ه"])
        elif word[len(word) - 1] == "a":
            listm.append(["ا"])
        elif word[len(word) - 1] == "o":
            listm.append(["و"])
        elif word[len(word) - 1] == "u":
            listm.append(["و"])

        return listm

    def map_switcher(self, ch):
        switcher = {
            "a": None,
            "e": None,
            "o": None,
            "u": None,
            "ee": None,

            "ou": None
        }
        return switcher.get(ch, ch)

    def make_word(self, chp):
        word_list = [[]]
        for char in chp:
            word_list_temp = []
            for tmp_word_list in word_list:
                for chch in char:
                    tmp = copy.deepcopy(tmp_word_list)
                    tmp.append(chch)
                    word_list_temp.append(tmp)
            word_list = word_list_temp
        return word_list

    def escalation(self, word):
        tmp = []
        i = 0
        t = len(word)
        while i < t - 1:
            tmp.append(word[i])
            if word[i] == word[i + 1]:
                i += 1
            i += 1
        if i != t:
            tmp.append(word[i])
        return tmp

    def char_switcher(self, ch):
        switcher = {
            'a': ["", "ا"],
            'c': ["ث", "ص", "ص"],
            'h': ["ه", "ح"],
            'b': ["ب"],
            'p': ["پ"],
            't': ["ت", "ط"],
            's': ["س", "ص", "ث"],
            'j': ["ج"],
            'ch': ["چ"],
            'kh': ["خ"],
            'q': ["ق", "غ"],
            'd': ["د"],
            'z': ["ز", "ذ", "ض", "ظ"],
            'r': ["ر"],
            'zh': ["ژ"],
            'sh': ["ش"],
            'gh': [",ق", "غ"],
            'f': ["ف"],
            'k': ["ک"],
            'g': ["گ"],
            'l': ["ل"],
            'm': ["م"],
            'n': ["ن"],
            'v': ["و"],
            'aa': ["ا"],
            'ee': ["ی"],
            'oo': ["و"],
            'ou': ["و"],
            'i': ["ی"],
            'y': ["ی"],
            ' ': [""],
            'w': ["و"],
            'e': ["", "ه"],
            'o': ["", "و"]
        }
        return switcher.get(ch, "")


class DictatioinNormalizer:
    def __init__(self, zwnj_database_path='./Parsivar/Parsivar/resource/normalizer/N_cctt.txt',
                 compound_table_path='./Parsivar/Parsivar/resource/normalizer/Normalizer_WrongCompound.txt'):
        self.zwnj_table, self.zwnj_type = self.load_zwnj_database(zwnj_database_path)
        self.compound_table = self.load_compound_table(compound_table_path)
        look_up_t_farsi_str = './Parsivar/g2p_resources/look_up_t_farsi_extra_of_ariana_b_punch_EN_extra_of_nevisa.pkl'
        homograph_t_str = './Parsivar/g2p_resources/list_homograph.pkl'
        with open(look_up_t_farsi_str, 'rb') as f:
            self.look_up_t_farsi = pickle.load(f)
        with open(homograph_t_str, 'rb') as f:
            self.homograph = pickle.load(f)
        self.half_space_data = [half_space_group for half_space_group in self.look_up_t_farsi
                                if isinstance(half_space_group, str) and re.findall('\u200c', half_space_group)] + \
                               [half_space_group for half_space_group in self.homograph
                                if isinstance(half_space_group, str) and re.findall('\u200c', half_space_group)]
        self.half_space_data_splited = [word_group.split('\u200c') for word_group in self.half_space_data]
        self.half_space_data_joined = dict()
        for i in self.half_space_data_splited:
            for j in range(len(i)):
                for k in range(j + 1, len(i)):
                    self.half_space_data_joined["".join(i[j:k + 1])] = '\u200c'.join(i[j:k + 1])

    def remove_extra_space_zwnj(self, doc_string):
        extra_regex = re.compile(r'([\u0648\u0624\u062f\u0630\u0631\u0632\u0698\u0627'
                                 r'\u0622\u0654\u0621\u0623\u0625\u0060][\u0627\u064b][\u0621\u064b])(\u200c)')
        doc_string = extra_regex.sub('\1', doc_string)
        extra_regex = re.compile(r'([\u0030-\u0039])(\u200c)')
        doc_string = extra_regex.sub('\1', doc_string)
        extra_regex = re.compile(r'(\u200c)([\u0030-\u0039])')
        doc_string = extra_regex.sub('\2', doc_string)
        extra_regex = re.compile(r'([a-z|A-Z])(\u200c)')
        doc_string = extra_regex.sub('\1', doc_string)
        extra_regex = re.compile(r'(\u200c)([a-z|A-Z])')
        doc_string = extra_regex.sub('\2', doc_string)
        extra_regex = re.compile(r'[\s]{2,}')
        doc_string = extra_regex.sub(' ', doc_string)
        extra_regex = re.compile(r'[\u200c]{2,}')
        doc_string = extra_regex.sub('\u200c', doc_string)
        extra_regex = re.compile(r'\u200c\s')
        doc_string = extra_regex.sub(' ', doc_string)
        extra_regex = re.compile(r'\s\u200c')
        doc_string = extra_regex.sub(' ', doc_string)
        return doc_string

    def correct_compound(self, doc_string, neighborhood):
        word_list = doc_string.split()
        if neighborhood > len(word_list):
            neighborhood = len(word_list)
        output_list = []
        i = 0
        while i < len(word_list):
            for j in range(neighborhood, 0, -1):
                if i + j < len(word_list):
                    comp = " ".join(word_list[i:i + j])
                else:
                    comp = " ".join(word_list[i:])
                if comp in self.compound_table:
                    output_list.append(self.compound_table[comp])
                    i += j
                    break
            else:
                output_list.append(word_list[i])
                i += 1
        return " ".join(output_list)

    def half_space_corrector(self, string):
        word_list = string.split()
        word_list = [word.strip('\u200c') for word in word_list]
        word_index = 0
        result = []
        while word_index < len(word_list):
            a=1
            if word_list[word_index] in self.half_space_data_joined and \
                    word_list[word_index] not in self.look_up_t_farsi and word_list[word_index] not in self.homograph:
                word_list[word_index] = self.half_space_data_joined[word_list[word_index]]
            possible_words = [i for i in self.half_space_data_splited if word_list[word_index] == i[0]
                              and word_index + 1 < len(word_list) and word_list[word_index + 1] == i[1]]
            best_words = [word_list[word_index]]
            max_match = 0
            if possible_words:
                for j in possible_words:
                    for k in range(len(j)):
                        if k >= len(word_list):
                            break
                        if len(word_list) <= word_index+k or j[k] != word_list[word_index + k]:
                            break
                        if max_match < k:
                            max_match = k
                            best_words = j[:k + 1]

            else:
                if word_index + 1 < len(word_list) and (word_list[word_index + 1] in
                                                        ['ام', 'ات', 'اش', 'مان', 'شان', 'تان', 'ايم', 'ايد', 'اند',
                                                         'اي', 'ها', 'هاي', 'هايي', 'هايم', 'هايشان', 'هايتان',
                                                         'هايمان', 'هايت', 'هايش', 'هائي' 'تر', 'ترين', 'ي', 'يي']
                                                        or word_list[word_index] in ['مي', 'نمي', 'برنمي', 'درمي',
                                                                                     'درنمي']):
                    best_words = [word_list[word_index], word_list[word_index + 1]]
                    max_match = 1
            result.append('\u200c'.join(best_words))
            word_index += max_match + 1
        result = " ".join(result)
        return result

    def hamze_corrector(self, string):
        word_list = string.split()
        for word in word_list:
            if word not in self.homograph + self.look_up_t_farsi:
                pass

    def load_zwnj_database(self, zwnj_database_file):
        table_words = dict()
        words_type = dict()
        with open(zwnj_database_file, encoding='utf-8') as f:
            for line in f:
                cols = line.split('\t')
                connection_type = cols[0]
                incorrect_word = cols[1]
                correct_word = cols[2].replace(' ', '_')
                table_words[incorrect_word.replace('\u200c', '^')] = correct_word
                if connection_type in ["مجزا", "به قبلي", "به بعدي"]:
                    words_type[correct_word.replace('\u200c', '^')] = connection_type
        return table_words, words_type

    def is_english_grapheme(self, char):
        return 'a' < char < 'z' or 'A' < char < 'Z'

    def is_english_word(self, word):
        return self.is_english_grapheme(word[0])

    def is_punctuation(self, char):
        return char in ['-', '=', '~', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '[', '{', ']', '}',
                        ':', ';', '\'', '\"', ',', '<', '.', '>', '/', '?', '|', '\\', '÷', '٬', '٫', '٪', '×', '،',
                        '»', '«', '؛', '؟', '…', '”', 'ˈ']

    def join_words_without_rules(self, docstring):
        words = docstring.split()
        correct_docstring = []
        for ind in range(len(words)):
            word = words[ind]
            table_word = word.replace('\u200c', '^')
            if table_word in self.zwnj_type.keys():
                if self.zwnj_type[table_word] == "مجزا":
                    correct_docstring.append(self.zwnj_table[table_word].replace('_', ' '))
                elif self.zwnj_type[table_word] == "به قبلي":
                    if correct_docstring and not self.is_english_word(correct_docstring[-1]) \
                            and not self.is_punctuation(correct_docstring[-1][-1]):
                        correct_docstring.append('\u200c' + self.zwnj_table[table_word])
                    else:
                        correct_docstring.append(self.zwnj_table[table_word])
                elif self.zwnj_type[table_word] == "به بعدي":
                    if ind < len(words) - 1 and not self.is_english_word(words[ind + 1]) \
                            and not self.is_punctuation(words[ind + 1][0]):
                        correct_docstring.append(self.zwnj_table[table_word] + '\u200c')
                    else:
                        correct_docstring.append(self.zwnj_table[table_word])
            else:
                correct_docstring.append(word)
        return " ".join(correct_docstring)

    def load_compound_table(self, compound_table_path):
        compound_table = dict()
        with open(compound_table_path, encoding='utf-8') as f:
            for line in f:
                firstcol, secondcol = line.split('\t')
                compound_table[firstcol] = secondcol
        return compound_table


class TimeNormalizer:
    def __init__(self, spanning=2):
        self.numberNorm = NumberNormalizer()
        self.spanning = spanning

    def normalize_time(self, doc_string):
        time_regex = re.compile(r'([0-9]|0[0-9]|1[0-9]|2[0-4])[\s]*([:\-])[\s]*([0-5][\d]|60)[\']?'
                                r'[\s]*(([:\-])[\s]*([0-5][\d]|60)[\"]?)?')
        keywords_time1 = ['وقت', 'مدت', 'زمان']
        keywords_time2 = ['عصر', 'شب', 'غروب', 'ظهر', 'صبح', 'ساعت', 'بامداد']

        while time_regex.search(doc_string):
            match = time_regex.search(doc_string)
            if ngram_lookup(doc_string, match.start(), match.end(), keywords_time1, self.spanning) and \
                    not ngram_lookup(doc_string, match.start(), match.end(), keywords_time2, self.spanning):
                if match[6] is not None:
                    hour = int(match[1])
                    minute = int(match[3])
                    second = int(match[6])
                    doc_string = self.time_duration_to_text_sec(doc_string, match.start(), match.end(),
                                                                hour, minute, second)
                else:
                    hour = int(match[1])
                    minute = int(match[3])
                    doc_string = self.time_duration_to_text(doc_string, match.start(), match.end(), hour, minute)
            else:
                if match[6] is not None:
                    hour = int(match[1])
                    minute = int(match[3])
                    second = int(match[6])
                    doc_string = self.time_to_text_sec(doc_string, match.start(), match.end(), hour, minute, second)
                else:
                    hour = int(match[1])
                    minute = int(match[3])
                    doc_string = self.time_to_text(doc_string, match.start(), match.end(), hour, minute)

        return doc_string

    def time_duration_to_text_sec(self, doc_string, start, end, hour, minute, second):
        string = self.numberNorm.convert(hour) + ' ساعت وَ ' + self.numberNorm.convert(minute) + ' دقیقه وَ ' + \
                 self.numberNorm.convert(second) + ' ثانیه'
        doc_string = doc_string[:start] + string + doc_string[end:]
        return doc_string

    def time_duration_to_text(self, doc_string, start, end, hour, minute):
        string = self.numberNorm.convert(hour) + ' ساعت وَ ' + self.numberNorm.convert(minute) + ' دقیقه '
        doc_string = doc_string[:start] + string + doc_string[end:]
        return doc_string

    def time_to_text_sec(self, doc_string, start, end, hour, minute, second):
        string = self.numberNorm.convert(hour) + ' وَ ' + self.numberNorm.convert(minute) + ' وَ ' + \
                 self.numberNorm.convert(second) + ' ثانیه '
        doc_string = doc_string[:start] + string + doc_string[end:]
        return doc_string

    def time_to_text(self, doc_string, start, end, hour, minute):
        string = self.numberNorm.convert(hour) + ' وَ' \
                                                 ' ' + self.numberNorm.convert(minute) + ' دقیقه '
        doc_string = doc_string[:start] + string + doc_string[end:]
        return doc_string


def ngram_lookup(doc_string, start, end, word_list, span):
    result = False
    for spanns in range(1, span):
        if any(x in word_list for x in doc_string[:start].rsplit(maxsplit=spanns + 1)[-spanns:]):
            result = True
        if any(x in word_list for x in doc_string[end:].split(maxsplit=spanns + 1)[:spanns]):
            result = True
    return result
