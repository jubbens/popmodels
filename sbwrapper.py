from SeqBreed import genome as gg
import numpy as np
import warnings


class GFounderMock:
    """A mock object for GFounder in the case of a random population"""
    def __init__(self, nsnp, nbase, ploidy=2):
        self.nbase = nbase
        self.ploidy = ploidy
        self.nsnp = nsnp
        self.g = np.random.randint(0, 2, (nsnp, nbase * ploidy), dtype=np.uint8)

        self.f = self.g.mean(axis=1)
        self.f[np.where(self.f > 0.5)] = 1. - self.f[np.where(self.f > 0.5)]


class GenomeMock:
    """A mock object for Genome in the case of a random population"""
    def __init__(self, num_snp, nchr, ploidy=2, autopolyploid=True):
        self.nchr = nchr
        self.num_snp = num_snp
        self.ploidy = ploidy
        self.autopolyploid = autopolyploid

        chr_names = [str(i) for i in range(nchr)]
        self.dictionar = dict(zip(chr_names, range(self.nchr)))

        snp_per_chrom = int(num_snp / nchr)
        bounds = np.arange(0, num_snp, snp_per_chrom)

        self.chrs = [gg.Chromosome(name=chr_names[i], pos=bounds[i], nsnp=snp_per_chrom, length=snp_per_chrom,
                                   xchr=False, ychr=False, mtchr=False)
                     for i in range(nchr)]

        nsnp = list(self.chrs[i].nsnp for i in range(self.nchr))
        self.cumpos = np.cumsum(nsnp) - nsnp


class Individual:
    def __init__(self):
        self.id = None
        self.phenotypes = []
        self.genotype = None
        self.dam = None
        self.sire = None


class Trial:
    def __init__(self):
        self._generation_bounds = []
        self._pop = None
        self._gbase = None
        self._gfeatures = None
        self._traits = None
        self._pedfile = None
        self._seqfile = None
        self._ploidy = None

    def get_generation(self, gen_num):
        bounds = self._generation_bounds[gen_num]

        if self._seqfile is None:
            chipseq = gg.Chip(gfounders=self._gbase, genome=self._gfeatures, nsnp=self._gfeatures.num_snp, name='seq_chip')
        else:
            chipseq = gg.Chip(chipFile=self._seqfile, genome=self._gfeatures, name='seq_chip')

        X = gg.do_X(self._pop.inds, self._gfeatures, self._gbase, chip=chipseq)

        gen = []

        for i in range(bounds[0], bounds[1]):
            ind = Individual()
            ind.id = self._pop.inds[i].id
            ind.phenotypes = self._pop.inds[i].y
            ind.genotype = X[:, i]

            ind.dam = self._pop.inds[i].id_dam
            ind.sire = self._pop.inds[i].id_sire

            gen.append(ind)

        return gen

    def import_founder_data(self, genfile, ploidy, pedfile=None, seqfile=None):
        self._gbase = gg.GFounder(vcfFile=genfile, snpFile=seqfile, ploidy=ploidy)
        self._gfeatures = gg.Genome(snpFile=seqfile, ploidy=ploidy)
        self._pedfile = pedfile
        self._seqfile = seqfile
        self._ploidy = ploidy

    def generate_random_founders(self, nsnp, nbase, ploidy=2, nchrom=24):
        self._gbase = GFounderMock(nsnp, nbase, ploidy=ploidy)
        self._gfeatures = GenomeMock(nsnp, nchrom, ploidy=ploidy)
        self._pedfile = None
        self._seqfile = None
        self._ploidy = ploidy

    def define_traits(self, h2, nqtl=None, qtl_file=None):
        if nqtl is not None:
            self._traits = gg.QTNs(h2=h2, genome=self._gfeatures, nqtn=nqtl)
        elif qtl_file is not None:
            self._traits = gg.QTNs(h2=h2, genome=self._gfeatures, qtnFile=qtl_file)
        else:
            warnings.warn('Need either nqtl or qtl_file specified.')
            exit()

    def make_founder_generation(self):
        self._pop = gg.Population(self._gfeatures, pedFile=self._pedfile, generation=None, qtns=self._traits, gfounders=self._gbase)

        self._generation_bounds.append((0, self._pop.n))

    def make_crosses(self, crosses, num_children):
        low = self._generation_bounds[-1][1]

        if isinstance(num_children, int):
            # We are doing a constant number of children
            high = low + (len(crosses) * num_children)
            self._generation_bounds.append((low, high))

            num_children = range(num_children)

            for cross in crosses:
                for i in range(num_children):
                    parents = [self._pop.inds[cross[0] - 1], self._pop.inds[cross[1] - 1]]  # -1 because this is zero-indexed
                    self._pop.addInd(parents, genome=self._gfeatures, gfounders=self._gbase, qtns=self._traits, id=None, sex=None, t=None)
        else:
            # We are doing a variable number of children per cross
            high = low + np.sum(num_children)
            self._generation_bounds.append((low, high))

            for cross, nc in zip(crosses, num_children):
                for i in range(nc):
                    parents = [self._pop.inds[cross[0] - 1], self._pop.inds[cross[1] - 1]]  # -1 because this is zero-indexed
                    self._pop.addInd(parents, genome=self._gfeatures, gfounders=self._gbase, qtns=self._traits, id=None, sex=None, t=None)
