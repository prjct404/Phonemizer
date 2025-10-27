import nltk

class FindChunks():
    def __init__(self):
        self.grammar = r"""
                        VP: {<ADJ_SIM><V_PRS>}
                        VP: {<ADJ_INO><V.*>}
                        VP: {<V_PRS><N_SING><V_SUB>}
                        NP: {<N_SING><ADJ.*><N_SING>}
                        NP: {<N.*><PRO>}
                        VP: {<N_SING><V_.*>}
                        VP: {<V.*>+}
                        NP: {<ADJ.*>?<N.*>+ <ADJ.*>?}
                        DNP: {<DET><NP>}
                        PP: {<ADJ_CMPR><P>}
                        PP: {<ADJ_SIM><P>}
                        PP: {<P><N_SING>}
                        PP: {<P>*}
                        DDNP: {<NP><DNP>}
                        NPP: {<PP><NP>+}
                        """
