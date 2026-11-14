"""Verified reference database (IOP numeric style).

Every entry was checked against at least one primary database record (Crossref,
a publisher/proceedings page, an arXiv/PMLR record, an author manuscript, or a
library catalogue).  'v' records how: 'C' = Crossref record, 'P' = publisher or
proceedings page, 'A' = arXiv/PMLR/HAL record, 'M' = full text or manuscript
read, 'L' = library catalogue (books), 'W' = web search of authoritative
listing.  'content' records whether the content that the paper relies on was
read ('full', 'abstract', 'title', 'none').
"""

R = {}


def add(key, kind, authors, year, title, venue='', vol='', pages='', doi='', v='', content='title', extra=''):
    R[key] = dict(kind=kind, authors=authors, year=year, title=title, venue=venue, vol=vol,
                  pages=pages, doi=doi, v=v, content=content, extra=extra)


# ---------------------------------------------------------------- reviews of PHM / RUL
add('si2011', 'art', 'Si X-S, Wang W, Hu C-H and Zhou D-H', 2011,
    'Remaining useful life estimation -- a review on the statistical data driven approaches',
    'Eur. J. Oper. Res.', '213', '1--14', '10.1016/j.ejor.2010.11.018', 'C', 'title')
add('lei2018', 'art', 'Lei Y, Li N, Guo L, Li N, Yan T and Lin J', 2018,
    'Machinery health prognostics: a systematic review from data acquisition to RUL prediction',
    'Mech. Syst. Signal Process.', '104', '799--834', '10.1016/j.ymssp.2017.11.016', 'C', 'title')
add('zio2022', 'art', 'Zio E', 2022,
    'Prognostics and health management (PHM): where are we and where do we (need to) go in theory and practice',
    'Reliab. Eng. Syst. Saf.', '218', '108119', '10.1016/j.ress.2021.108119', 'C', 'title')
add('fink2020', 'art', 'Fink O, Wang Q, Svens\\\'en M, Dersin P, Lee W-J and Ducoffe M', 2020,
    'Potential, challenges and future directions for deep learning in prognostics and health management applications',
    'Eng. Appl. Artif. Intell.', '92', '103678', '10.1016/j.engappai.2020.103678', 'C', 'title')
add('javed2017', 'art', 'Javed K, Gouriveau R and Zerhouni N', 2017,
    'State of the art and taxonomy of prognostics approaches, trends of prognostics applications and open issues towards maturity at different technology readiness levels',
    'Mech. Syst. Signal Process.', '94', '214--36', '10.1016/j.ymssp.2017.01.050', 'C', 'title')
add('liao2014', 'art', 'Liao L and K\\"ottig F', 2014,
    'Review of hybrid prognostics approaches for remaining useful life prediction of engineered systems, and an application to battery life prediction',
    'IEEE Trans. Reliab.', '63', '191--207', '10.1109/TR.2014.2299152', 'C', 'title')
add('zhang2018', 'art', 'Zhang Z, Si X, Hu C and Lei Y', 2018,
    'Degradation data analysis and remaining useful life estimation: a review on Wiener-process-based methods',
    'Eur. J. Oper. Res.', '271', '775--96', '10.1016/j.ejor.2018.02.033', 'C', 'title')
add('saxena2008metrics', 'proc', 'Saxena A, Celaya J, Balaban E, Goebel K, Saha B, Saha S and Schwabacher M', 2008,
    'Metrics for evaluating performance of prognostic techniques',
    'Proc. Int. Conf. on Prognostics and Health Management', '', '1--17', '10.1109/PHM.2008.4711436', 'C', 'title')

# ---------------------------------------------------------------- stochastic degradation / first hitting time
add('whitmore1995', 'art', 'Whitmore G A', 1995,
    'Estimating degradation by a Wiener diffusion process subject to measurement error',
    'Lifetime Data Anal.', '1', '307--19', '10.1007/BF00985762', 'C', 'title')
add('lee2006', 'art', 'Lee M-L T and Whitmore G A', 2006,
    'Threshold regression for survival analysis: modeling event times by a stochastic process reaching a boundary',
    'Stat. Sci.', '21', '501--13', '10.1214/088342306000000330', 'C', 'title')
add('si2012', 'art', 'Si X-S, Wang W, Hu C-H, Zhou D-H and Pecht M G', 2012,
    'Remaining useful life estimation based on a nonlinear diffusion degradation process',
    'IEEE Trans. Reliab.', '61', '50--67', '10.1109/TR.2011.2182221', 'C', 'abstract')
add('gebraeel2005', 'art', 'Gebraeel N Z, Lawley M A, Li R and Ryan J K', 2005,
    'Residual-life distributions from component degradation signals: a Bayesian approach',
    'IIE Trans.', '37', '543--57', '10.1080/07408170590929018', 'C', 'title')
add('wang2010', 'art', 'Wang X', 2010,
    'Wiener processes with random effects for degradation data',
    'J. Multivariate Anal.', '101', '340--51', '10.1016/j.jmva.2008.12.007', 'C', 'abstract')
add('ye2015', 'art', 'Ye Z-S and Xie M', 2015,
    'Stochastic modelling and analysis of degradation for highly reliable products',
    'Appl. Stoch. Models Bus. Ind.', '31', '16--32', '10.1002/asmb.2063', 'C', 'title')
add('tang2014', 'art', 'Tang S, Yu C, Wang X, Guo X and Si X', 2014,
    'Remaining useful life prediction of lithium-ion batteries based on the Wiener process with measurement error',
    'Energies', '7', '520--47', '10.3390/en7020520', 'C', 'title')
add('medjaher2012', 'art', 'Medjaher K, Tobon-Mejia D A and Zerhouni N', 2012,
    'Remaining useful life estimation of critical components with application to bearings',
    'IEEE Trans. Reliab.', '61', '292--302', '10.1109/TR.2012.2194175', 'C', 'title')
add('chhikara1989', 'book', 'Chhikara R S and Folks J L', 1989,
    'The Inverse Gaussian Distribution: Theory, Methodology, and Applications',
    '(New York: Marcel Dekker)', '', '', '', 'L', 'none')
add('folks1978', 'art', 'Folks J L and Chhikara R S', 1978,
    'The inverse Gaussian distribution and its statistical application -- a review',
    'J. R. Stat. Soc. B', '40', '263--75', '10.1111/j.2517-6161.1978.tb01039.x', 'C', 'none')
add('tweedie1957', 'art', 'Tweedie M C K', 1957,
    'Statistical properties of inverse Gaussian distributions. I',
    'Ann. Math. Stat.', '28', '362--77', '10.1214/aoms/1177706964', 'C', 'none')
add('meeker1998', 'book', 'Meeker W Q and Escobar L A', 1998,
    'Statistical Methods for Reliability Data', '(New York: Wiley)', '', '', '', 'L', 'none')

# ---------------------------------------------------------------- health-indicator construction
add('coble2009', 'proc', 'Coble J and Hines J W', 2009,
    'Identifying optimal prognostic parameters from data: a genetic algorithms approach',
    'Annu. Conf. of the Prognostics and Health Management Society', '1', '', '', 'P', 'full')
add('javed2013', 'proc', 'Javed K, Gouriveau R, Zerhouni N and Nectoux P', 2013,
    'A feature extraction procedure based on trigonometric functions and cumulative descriptors to enhance prognostics modeling',
    'Proc. IEEE Conf. on Prognostics and Health Management', '', '1--7', '10.1109/ICPHM.2013.6621413', 'C', 'full')
add('zhang2016', 'art', 'Zhang B, Zhang L and Xu J', 2016,
    'Degradation feature selection for remaining useful life prediction of rolling element bearings',
    'Qual. Reliab. Eng. Int.', '32', '547--54', '10.1002/qre.1771', 'C', 'abstract')
add('qian2015', 'proc', 'Qian F and Niu G', 2015,
    'Remaining useful life prediction using ranking mutual information based monotonic health indicator',
    'Proc. Prognostics and System Health Management Conf.', '', '1--5', '10.1109/PHM.2015.7380042', 'C', 'title')
add('guo2017', 'art', 'Guo L, Li N, Jia F, Lei Y and Lin J', 2017,
    'A recurrent neural network based health indicator for remaining useful life prediction of bearings',
    'Neurocomputing', '240', '98--109', '10.1016/j.neucom.2017.02.045', 'C', 'title')
add('li2018', 'art', 'Li X, Ding Q and Sun J-Q', 2018,
    'Remaining useful life estimation in prognostics using deep convolution neural networks',
    'Reliab. Eng. Syst. Saf.', '172', '1--11', '10.1016/j.ress.2017.11.021', 'C', 'title')
add('severson2019', 'art', 'Severson K A et al', 2019,
    'Data-driven prediction of battery cycle life before capacity degradation',
    'Nat. Energy', '4', '383--91', '10.1038/s41560-019-0356-8', 'C', 'title')
add('attia2020', 'art', 'Attia P M et al', 2020,
    'Closed-loop optimization of fast-charging protocols for batteries with machine learning',
    'Nature', '578', '397--402', '10.1038/s41586-020-1994-5', 'C', 'title')
add('ma2022', 'art', 'Ma G et al', 2022,
    'Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning',
    'Energy Environ. Sci.', '15', '4083--94', '10.1039/D2EE01676A', 'C', 'title')
add('tan2025', 'proc', 'Tan R et al', 2025,
    'BatteryLife: a comprehensive dataset and benchmark for battery life prediction',
    'Proc. 31st ACM SIGKDD Conf. on Knowledge Discovery and Data Mining', '2', '5789--800', '10.1145/3711896.3737372', 'C', 'abstract')
add('zhu2022', 'art', 'Zhu J et al', 2022,
    'Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation',
    'Nat. Commun.', '13', '2261', '10.1038/s41467-022-29837-w', 'C', 'title')
add('mauthe2024', 'arxiv', 'Mauthe F, Braun C, Raible J, Zeiler P and Huber M F', 2024,
    'Overview of publicly available degradation data sets for tasks within prognostics and health management',
    'arXiv preprint', '', '', '', 'A', 'full', extra='arXiv:2403.13694')

# ---------------------------------------------------------------- uncertainty
add('sankararaman2013', 'proc', 'Sankararaman S and Goebel K', 2013,
    'Why is the remaining useful life prediction uncertain?',
    'Annu. Conf. of the Prognostics and Health Management Society', '5', '', '10.36001/phmconf.2013.v5i1.2263', 'C', 'title')
add('sankararaman2015', 'art', 'Sankararaman S', 2015,
    'Significance, interpretation, and quantification of uncertainty in prognostics and remaining useful life prediction',
    'Mech. Syst. Signal Process.', '52--53', '228--47', '10.1016/j.ymssp.2014.05.029', 'C', 'title')
add('hullermeier2021', 'art', 'H\\"ullermeier E and Waegeman W', 2021,
    'Aleatoric and epistemic uncertainty in machine learning: an introduction to concepts and methods',
    'Mach. Learn.', '110', '457--506', '10.1007/s10994-021-05946-3', 'C', 'title')
add('kendall2017', 'proc', 'Kendall A and Gal Y', 2017,
    'What uncertainties do we need in Bayesian deep learning for computer vision?',
    'Advances in Neural Information Processing Systems', '30', '', '', 'A', 'title', extra='arXiv:1703.04977')
add('lakshminarayanan2017', 'proc', 'Lakshminarayanan B, Pritzel A and Blundell C', 2017,
    'Simple and scalable predictive uncertainty estimation using deep ensembles',
    'Advances in Neural Information Processing Systems', '30', '', '', 'A', 'title', extra='arXiv:1612.01474')

# ---------------------------------------------------------------- information bottleneck and representation learning
add('tishby2000', 'arxiv', 'Tishby N, Pereira F C and Bialek W', 2000,
    'The information bottleneck method', 'arXiv:physics/0004057', '', '', '', 'A', 'title')
add('alemi2017', 'proc', 'Alemi A A, Fischer I, Dillon J V and Murphy K', 2017,
    'Deep variational information bottleneck', 'Proc. Int. Conf. on Learning Representations', '', '', '', 'A', 'title', extra='arXiv:1612.00410')
add('achille2018', 'art', 'Achille A and Soatto S', 2018,
    'Emergence of invariance and disentanglement in deep representations',
    'J. Mach. Learn. Res.', '19', '1--34', '', 'P', 'abstract')
add('saxe2019', 'art', 'Saxe A M, Bansal Y, Dapello J, Advani M, Kolchinsky A, Tracey B D and Cox D D', 2019,
    'On the information bottleneck theory of deep learning',
    'J. Stat. Mech.: Theory Exp.', '2019', '124020', '10.1088/1742-5468/ab3985', 'C', 'title')
add('kolchinsky2019', 'art', 'Kolchinsky A, Tracey B D and Wolpert D H', 2019,
    'Nonlinear information bottleneck', 'Entropy', '21', '1181', '10.3390/e21121181', 'C', 'title')
add('poole2019', 'proc', 'Poole B, Ozair S, van den Oord A, Alemi A A and Tucker G', 2019,
    'On variational bounds of mutual information', 'Proc. Int. Conf. on Machine Learning (PMLR vol 97)', '', '5171--80', '', 'A', 'title')
add('kingma2014', 'proc', 'Kingma D P and Welling M', 2014,
    'Auto-encoding variational Bayes', 'Proc. Int. Conf. on Learning Representations', '', '', '', 'A', 'title', extra='arXiv:1312.6114')
add('higgins2017', 'proc', 'Higgins I, Matthey L, Pal A, Burgess C, Glorot X, Botvinick M, Mohamed S and Lerchner A', 2017,
    '$\\beta$-VAE: learning basic visual concepts with a constrained variational framework',
    'Proc. Int. Conf. on Learning Representations', '', '', '', 'W', 'title')

# ---------------------------------------------------------------- information theory and estimation bounds
add('shannon1948', 'art', 'Shannon C E', 1948, 'A mathematical theory of communication',
    'Bell Syst. Tech. J.', '27', '379--423, 623--56', '10.1002/j.1538-7305.1948.tb01338.x', 'C', 'none')
add('cover2006', 'book', 'Cover T M and Thomas J A', 2006, 'Elements of Information Theory 2nd edn',
    '(Hoboken, NJ: Wiley-Interscience)', '', '', '', 'L', 'none')
add('stam1959', 'art', 'Stam A J', 1959,
    'Some inequalities satisfied by the quantities of information of Fisher and Shannon',
    'Inf. Control', '2', '101--12', '10.1016/S0019-9958(59)90348-1', 'C', 'none')
add('vantrees1968', 'book', 'Van Trees H L', 1968, 'Detection, Estimation, and Modulation Theory, Part I',
    '(New York: Wiley)', '', '', '', 'L', 'none')
add('gill1995', 'art', 'Gill R D and Levit B Y', 1995, 'Applications of the van Trees inequality: a Bayesian Cram\\\'er--Rao bound',
    'Bernoulli', '1', '59--79', '10.2307/3318681', 'P', 'abstract')
add('tichavsky1998', 'art', 'Tichavsk\\\'y P, Muravchik C H and Nehorai A', 1998,
    'Posterior Cram\\\'er--Rao bounds for discrete-time nonlinear filtering',
    'IEEE Trans. Signal Process.', '46', '1386--96', '10.1109/78.668800', 'C', 'title')
add('vasicek1976', 'art', 'Vasicek O', 1976, 'A test for normality based on sample entropy',
    'J. R. Stat. Soc. B', '38', '54--9', '10.1111/j.2517-6161.1976.tb01566.x', 'C', 'none')
add('kraskov2004', 'art', 'Kraskov A, St\\"ogbauer H and Grassberger P', 2004, 'Estimating mutual information',
    'Phys. Rev. E', '69', '066138', '10.1103/PhysRevE.69.066138', 'C', 'none')
add('paninski2003', 'art', 'Paninski L', 2003, 'Estimation of entropy and mutual information',
    'Neural Comput.', '15', '1191--253', '10.1162/089976603321780272', 'C', 'none')
add('mcallester2020', 'proc', 'McAllester D and Stratos K', 2020, 'Formal limitations on the measurement of mutual information',
    'Proc. Int. Conf. on Artificial Intelligence and Statistics (PMLR vol 108)', '', '875--84', '', 'A', 'abstract')

# ---------------------------------------------------------------- statistics, filtering, forecasting
add('bishop2006', 'book', 'Bishop C M', 2006, 'Pattern Recognition and Machine Learning', '(New York: Springer)', '', '', '', 'L', 'none')
add('kalman1960', 'art', 'Kalman R E', 1960, 'A new approach to linear filtering and prediction problems',
    'J. Basic Eng.', '82', '35--45', '10.1115/1.3662552', 'C', 'none')
add('harvey1989', 'book', 'Harvey A C', 1989, 'Forecasting, Structural Time Series Models and the Kalman Filter',
    '(Cambridge: Cambridge University Press)', '', '', '', 'L', 'none')
add('durbin2012', 'book', 'Durbin J and Koopman S J', 2012, 'Time Series Analysis by State Space Methods 2nd edn',
    '(Oxford: Oxford University Press)', '', '', '', 'L', 'none')
add('genest1986', 'art', 'Genest C and Zidek J V', 1986, 'Combining probability distributions: a critique and an annotated bibliography',
    'Stat. Sci.', '1', '114--35', '10.1214/ss/1177013825', 'C', 'none')
add('hinton2002', 'art', 'Hinton G E', 2002, 'Training products of experts by minimizing contrastive divergence',
    'Neural Comput.', '14', '1771--800', '10.1162/089976602760128018', 'C', 'none')
add('gneiting2007', 'art', 'Gneiting T and Raftery A E', 2007, 'Strictly proper scoring rules, prediction, and estimation',
    'J. Am. Stat. Assoc.', '102', '359--78', '10.1198/016214506000001437', 'C', 'none')
add('efron1993', 'book', 'Efron B and Tibshirani R J', 1993, 'An Introduction to the Bootstrap',
    '(New York: Chapman and Hall)', '', '', '', 'L', 'none')
add('welch1947', 'art', 'Welch B L', 1947, 'The generalization of `Student\'s\' problem when several different population variances are involved',
    'Biometrika', '34', '28--35', '10.1093/biomet/34.1-2.28', 'C', 'none')
add('bouthillier2021', 'art', 'Bouthillier X et al', 2021, 'Accounting for variance in machine learning benchmarks',
    'Proc. Mach. Learn. Syst.', '3', '747--69', '', 'P', 'title')
add('durrett2010', 'book', 'Durrett R', 2010, 'Probability: Theory and Examples 4th edn',
    '(Cambridge: Cambridge University Press)', '', '', '', 'L', 'none')

# ---------------------------------------------------------------- data sets
add('saha2007', 'data', 'Saha B and Goebel K', 2007, 'Battery data set',
    'NASA Prognostics Data Repository (Moffett Field, CA: NASA Ames Research Center)', '', '', '', 'P', 'none')
add('he2011', 'art', 'He W, Williard N, Osterman M and Pecht M', 2011,
    'Prognostics of lithium-ion batteries based on Dempster--Shafer theory and the Bayesian Monte Carlo method',
    'J. Power Sources', '196', '10314--21', '10.1016/j.jpowsour.2011.08.040', 'C', 'none')
add('nectoux2012', 'proc', 'Nectoux P, Gouriveau R, Medjaher K, Ramasso E, Chebel-Morello B, Zerhouni N and Varnier C', 2012,
    'PRONOSTIA: an experimental platform for bearings accelerated degradation tests',
    'Proc. IEEE Int. Conf. on Prognostics and Health Management (Denver, CO)', '', '1--8', '', 'M', 'abstract')
add('lee2007', 'data', 'Lee J, Qiu H, Yu G, Lin J and Rexnord Technical Services', 2007, 'Bearing data set',
    'IMS, University of Cincinnati, NASA Prognostics Data Repository (Moffett Field, CA: NASA Ames Research Center)', '', '', '', 'P', 'none')
add('qiu2006', 'art', 'Qiu H, Lee J, Lin J and Yu G', 2006,
    'Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics',
    'J. Sound Vib.', '289', '1066--90', '10.1016/j.jsv.2005.03.007', 'C', 'none')
add('saxena2008', 'proc', 'Saxena A, Goebel K, Simon D and Eklund N', 2008,
    'Damage propagation modeling for aircraft engine run-to-failure simulation',
    'Proc. Int. Conf. on Prognostics and Health Management (Denver, CO)', '', '1--9', '10.1109/PHM.2008.4711414', 'M', 'full')

# ---------------------------------------------------------------- JRSE context and the authors' own work
add('sun2025', 'art', 'Sun C-P', 2025, 'Reliability science and engineering: a catalyst for engineering innovation',
    'J. Reliab. Sci. Eng.', '1', '020201', '10.1088/3050-2454/adbf22', 'C', 'title')
add('aven2021', 'art', 'Aven T', 2021, 'The reliability science: its foundation and link to risk science and other sciences',
    'Reliab. Eng. Syst. Saf.', '215', '107863', '10.1016/j.ress.2021.107863', 'C', 'title')
add('wang2026jrse', 'art', 'Wang L, Chen S, Cai Z, Liu W and Ye Z', 2026,
    'Cross-cycle health indicator construction and state of health prediction of lithium-ion batteries by cross recurrence quantification analysis',
    'J. Reliab. Sci. Eng.', '2', '035302', '10.1088/3050-2454/ae8dd1', 'C', 'abstract')
add('gu2025', 'art', 'Gu H, Li Y, Jiang L and Luo Z', 2025,
    'A generalized approach for remaining useful life prediction considering the multi-source uncertainty of mechanical systems',
    'J. Reliab. Sci. Eng.', '1', '025301', '10.1088/3050-2454/add5fa', 'C', 'abstract')
add('bisoyi2026', 'art', 'Bisoyi S, Kumar Rathi A and Mahato S', 2026,
    'Reliability assessment of remaining useful life predictions of roller bearings using deep learning models',
    'J. Reliab. Sci. Eng.', '2', '025302', '10.1088/3050-2454/ae61b4', 'C', 'title')
add('le2026jestch', 'art', 'Le H H and Nguyen K-A', 2026,
    'Information-theoretic limits of health indicator construction for remaining useful life prediction',
    'Eng. Sci. Technol. Int. J.', '81', '102485', '10.1016/j.jestch.2026.102485', 'C', 'abstract')
add('le2026mst', 'art', 'Le H H and Nguyen K-A', 2026,
    'Generalization bounds and sample complexity for remaining useful life prediction from complete degradation trajectories',
    'Meas. Sci. Technol.', '37', '226203', '10.1088/1361-6501/ae7109', 'C', 'abstract')
add('le2026ein', 'art', 'Le H H and Nguyen K-A', 2026,
    'Charging phase health indicators for battery state-of-health estimation: a systematic comparison of CC, CV, and combined approaches under cross-battery validation',
    'Eksploat. Niezawodn. -- Maint. Reliab.', '28', '', '10.17531/ein/220211', 'C', 'abstract')
