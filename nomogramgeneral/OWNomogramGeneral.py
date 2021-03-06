"""
<name>General Nomogram</name>
<description>Nomogram viewer for any classifier.</description>
<icon>icons/Nomogram.svg</icon>
<contact>Grega Pompe (grega.pompe(@at@)gmail.com)</contact>
<priority>2500</priority>
"""

#
# Nomogram is a Orange widget for
# for visualization of the knowledge
# obtained with classifier
#
import Orange
import Orange.OrangeWidgets.OWGUI
from Orange.OrangeWidgets.OWWidget import *
from Orange.OrangeWidgets.Classify.OWNomogramGraph import *
from orngDataCaching import *
import itertools
from distutils.command.install import install
from scipy.weave.catalog import get_catalog
from Orange.orng.orngDataCaching import getCached

aproxZero = 0.0001

SHOW_BIGGEST = 100

def getStartingPoint(d, min):
    if d == 0:
        return min
    elif min<0:
        curr_num = numpy.arange(-min+d, step=d)
        curr_num = curr_num[len(curr_num)-1]
        curr_num = -curr_num
    elif min - d <= 0:
        curr_num = 0
    else:
        curr_num = numpy.arange(min-d, step=d)
        curr_num = curr_num[len(curr_num)-1]
    return curr_num

def getRounding(d):
    if d == 0:
        return 2
    rndFac = math.floor(math.log10(d));
    if rndFac<-2:
        rndFac = int(-rndFac)
    else:
        rndFac = 2
    return rndFac

def avg(l):
    return sum(l)/len(l)


class OWNomogramGeneral(OWWidget):
    settingsList = ["alignType", "verticalSpacing", "contType", "verticalSpacingContinuous", "yAxis", "probability", "confidence_check", "confidence_percent", "histogram", "histogram_size", "sort_type"]
    contextHandlers = {"": DomainContextHandler("", ["TargetClassIndex"], matchValues=1)}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Nomogram", 1)

        #self.setWFlags(Qt.WResizeNoErase | Qt.WRepaintNoErase) #this works like magic.. no flicker during repaint!
        self.parent = parent
#        self.setWFlags(self.getWFlags()+Qt.WStyle_Maximize)

        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.alignType = 0
        self.contType = 0
        self.yAxis = 0
        self.probability = 0
        self.verticalSpacing = 60
        self.verticalSpacingContinuous = 100
        self.diff_between_ordinal = 30
        self.fontSize = 9
        self.lineWidth = 1
        self.histogram = 0
        self.histogram_size = 10
        self.data = None
        self.cl = None
        self.confidence_check = 0
        self.confidence_percent = 95
        self.sort_type = 0

        self.loadSettings()

        self.pointsName = ["Total", "Total"]
        self.totalPointsName = ["Probability", "Probability"]
        self.bnomogram = None


        self.inputs=[("Classifier", orange.Classifier, self.classifier), ("Data", Orange.data.Table, self.data)]


        self.TargetClassIndex = 0
        self.targetCombo = OWGUI.comboBox(self.controlArea, self, "TargetClassIndex", " Target Class ", addSpace=True, tooltip='Select target (prediction) class in the model.', callback = self.setTarget)

        self.alignRadio = OWGUI.radioButtonsInBox(self.controlArea, self,  'alignType', ['Align left', 'Align by zero influence'], box='Attribute placement',
                                                  tooltips=['Attributes in nomogram are left aligned', 'Attributes are not aligned, top scale represents true (normalized) regression coefficient value'],
                                                  addSpace=True,
                                                  callback=self.showNomogram)
        self.verticalSpacingLabel = OWGUI.spin(self.alignRadio, self, 'verticalSpacing', 15, 200, label = 'Vertical spacing:',  orientation = 0, tooltip='Define space (pixels) between adjacent attributes.', callback = self.showNomogram)

        self.ContRadio = OWGUI.radioButtonsInBox(self.controlArea, self, 'contType',   ['1D projection', '2D curve'], 'Continuous attributes',
                                tooltips=['Continuous attribute are presented on a single scale', 'Two dimensional space is used to present continuous attributes in nomogram.'],
                                addSpace=True,
                                callback=[lambda:self.verticalSpacingContLabel.setDisabled(not self.contType), self.showNomogram])

        self.verticalSpacingContLabel = OWGUI.spin(OWGUI.indentedBox(self.ContRadio, sep=OWGUI.checkButtonOffsetHint(self.ContRadio.buttons[-1])), self, 'verticalSpacingContinuous', 15, 200, label = "Height", orientation=0, tooltip='Define space (pixels) between adjacent 2d presentation of attributes.', callback = self.showNomogram)
        self.verticalSpacingContLabel.setDisabled(not self.contType)

        self.yAxisRadio = OWGUI.radioButtonsInBox(self.controlArea, self, 'yAxis', ['Point scale', 'Log odds ratios'], 'Scale',
                                tooltips=['values are normalized on a 0-100 point scale','values on top axis show log-linear contribution of attribute to full model'],
                                addSpace=True,
                                callback=self.showNomogram)

        layoutBox = OWGUI.widgetBox(self.controlArea, "Display", orientation=1, addSpace=True)

        self.probabilityCheck = OWGUI.checkBox(layoutBox, self, 'probability', 'Show prediction',  tooltip='', callback = self.setProbability)

        self.CICheck, self.CILabel = OWGUI.checkWithSpin(layoutBox, self, 'Confidence intervals (%):', min=1, max=99, step = 1, checked='confidence_check', value='confidence_percent', checkCallback=self.showNomogram, spinCallback = self.showNomogram)

        self.histogramCheck, self.histogramLabel = OWGUI.checkWithSpin(layoutBox, self, 'Show histogram, size', min=1, max=30, checked='histogram', value='histogram_size', step = 1, tooltip='-(TODO)-', checkCallback=self.showNomogram, spinCallback = self.showNomogram)

        OWGUI.separator(layoutBox)
        self.sortOptions = ["No sorting", "Absolute importance", "Positive influence", "Negative influence"]
        self.sortBox = OWGUI.comboBox(layoutBox, self, "sort_type", label="Sort by ", items=self.sortOptions, callback = self.sortNomogram, orientation="horizontal")


        OWGUI.rubber(self.controlArea)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.menuItemPrinter)



        #add a graph widget
        self.header = OWNomogramHeader(None, self.mainArea)
        self.header.setFixedHeight(60)
        self.header.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.header.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graph = OWNomogramGraph(self.bnomogram, self.mainArea)
        self.graph.setMinimumWidth(200)
        self.graph.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.footer = OWNomogramHeader(None, self.mainArea)
        self.footer.setFixedHeight(60*2+10)
        self.footer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.footer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.mainArea.layout().addWidget(self.header)
        self.mainArea.layout().addWidget(self.graph)
        self.mainArea.layout().addWidget(self.footer)
        self.resize(700,500)
        #self.repaint()
        #self.update()

        # mouse pressed flag
        self.mousepr = False

    def sendReport(self):
        if self.cl:
            tclass = self.cl.domain.classVar.values[self.TargetClassIndex]
        else:
            tclass = "N/A"
        self.reportSettings("Information",
                            [("Target class", tclass),
                             self.confidence_check and ("Confidence intervals", "%i%%" % self.confidence_percent),
                             ("Sorting", self.sortOptions[self.sort_type] if self.sort_type else "None")])
        if self.cl:
            canvases = header, graph, footer = self.header.scene(), self.graph.scene(), self.footer.scene()
            painter = QPainter()
            buffer = QPixmap(max(c.width() for c in canvases), sum(c.height() for c in canvases))
            painter.begin(buffer)
            painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
            header.render(painter, QRectF(0, 0, header.width(), header.height()), QRectF(0, 0, header.width(), header.height()))
            graph.render(painter, QRectF(0, header.height(), graph.width(), graph.height()), QRectF(0, 0, graph.width(), graph.height()))
            footer.render(painter, QRectF(0, header.height()+graph.height(), footer.width(), footer.height()), QRectF(0, 0, footer.width(), footer.height()))
            painter.end()
            self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))

        
    # Input channel: the Bayesian classifier
    def nbClassifier(self, cl):
        # this subroutine computes standard error of estimated beta. Note that it is used only for discrete data,
        # continuous data have a different computation.
        def errOld(e, priorError, key, data):
            inf = 0.0
            sume = e[0]+e[1]
            for d in data:
                if d[at]==key:
                    inf += (e[0]*e[1]/sume/sume)
            inf = max(inf, aproxZero)
            var = max(1/inf - priorError*priorError, 0)
            return (math.sqrt(var))

        def err(condDist, att, value, targetClass, priorError, data):
            sumE = sum(condDist)
            valueE = condDist[targetClass]
            distAtt = orange.Distribution(att, data)
            inf = distAtt[value]*(valueE/sumE)*(1-valueE/sumE)
            inf = max(inf, aproxZero)
            var = max(1/inf - priorError*priorError, 0)
            return (math.sqrt(var))

        classVal = cl.domain.classVar
        att = cl.domain.attributes

        # calculate prior probability
        dist1 = max(aproxZero, 1-cl.distribution[classVal[self.TargetClassIndex]])
        dist0 = max(aproxZero, cl.distribution[classVal[self.TargetClassIndex]])
        prior = dist0/dist1
        if self.data:
            sumd = dist1+dist0
            priorError = math.sqrt(1/((dist1*dist0/sumd/sumd)*len(self.data)))
        else:
            priorError = 0

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue("Constant", math.log(prior), error = priorError))
        else:
            self.bnomogram = BasicNomogram(self, AttValue("Constant", math.log(prior), error = priorError))

        if self.data:
            stat = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))

        for at in range(len(att)):
            a = None
            attribute = att[at]
            if attribute.varType == orange.VarTypes.Discrete:
                if attribute.ordered:
                    a = AttrLineOrdered(attribute.name, self.bnomogram)
                else:
                    a = AttrLine(attribute.name, self.bnomogram)
                for cd in cl.conditionalDistributions[at].keys():
                    conditionalDistribution = cl.conditionalDistributions[at][cd]
                    # calculuate thickness
                    targetClassDistribution = conditionalDistribution[classVal[self.TargetClassIndex]]
                    conditional0 = max(targetClassDistribution, aproxZero)
                    conditional1 = max(1-targetClassDistribution, aproxZero)
                    beta = math.log(conditional0/conditional1/prior)
                    

                    a.addAttValue(AttValue(cd, beta))

            else:
                a = AttrLineCont(attribute.name, self.bnomogram)
                numOfPartitions = 50

                if self.data:
                    maxAtValue = stat[at].max
                    minAtValue = stat[at].min
                else:
                    maxAtValue = cl.conditionalDistributions[at].keys()[len(cl.conditionalDistributions[at].keys())-1]
                    minAtValue = cl.conditionalDistributions[at].keys()[0]

                d = maxAtValue-minAtValue
                d = getDiff(d/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling
                curr_num = getStartingPoint(d, minAtValue)
                rndFac = getRounding(d)

                values = []
                for i in range(2*numOfPartitions):
                    if curr_num+i*d>=minAtValue and curr_num+i*d<=maxAtValue:
                        # get thickness
                        if self.data:
                            thickness = float(len(self.data.filter({attribute.name:(curr_num+i*d-d/2, curr_num+i*d+d/2)})))/len(self.data)
                        else:
                            thickness = 0.0
                        d_filter = filter(lambda x: x>curr_num+i*d-d/2 and x<curr_num+i*d+d/2, cl.conditionalDistributions[at].keys())
                        if len(d_filter)>0:
                            cd = cl.conditionalDistributions[at]
                            conditional0 = avg([cd[f][classVal[self.TargetClassIndex]] for f in d_filter])
                            conditional0 = min(1-aproxZero,max(aproxZero,conditional0))
                            conditional1 = 1-conditional0
                            try:
                                # compute error of loess in logistic space
                                var = avg([cd[f].variances[self.TargetClassIndex] for f in d_filter])
                                standard_error= math.sqrt(var)
                                rightError0 = (conditional0+standard_error)/max(conditional1-standard_error, aproxZero)
                                leftError0  =  max(conditional0-standard_error, aproxZero)/(conditional1+standard_error)
                                se = (math.log(rightError0) - math.log(leftError0))/2
                                se = math.sqrt(math.pow(se,2)+math.pow(priorError,2))

                                # add value to set of values
                                a.addAttValue(AttValue(str(round(curr_num+i*d,rndFac)),
                                                       math.log(conditional0/conditional1/prior),
                                                       lineWidth=thickness,
                                                       error = se))
                            except:
                                pass
                a.continuous = True
                # invert values:
            # if there are more than 1 value in the attribute, add it to the nomogram
            if a and len(a.attValues)>1:
                self.bnomogram.addAttribute(a)

        self.alignRadio.setDisabled(False)
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    # Input channel: the logistic regression classifier
    def lrClassifier(self, cl):
        if self.TargetClassIndex == 0 or self.TargetClassIndex == cl.domain.classVar[0]:
            mult = -1
        else:
            mult = 1

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue('Constant', mult*cl.beta[0], error = 0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue('Constant', mult*cl.beta[0], error = 0))

        # After applying feature subset selection on discrete attributes
        # aproximate unknown error for each attribute is math.sqrt(math.pow(cl.beta_se[0],2)/len(at))
        try:
            aprox_prior_error = math.sqrt(math.pow(cl.beta_se[0],2)/len(cl.domain.attributes))
        except:
            aprox_prior_error = 0

        domain = cl.continuizedDomain or cl.domain
        if domain:
            for at in domain.attributes:
                at.setattr("visited",0)

            for at in domain.attributes:
                if at.getValueFrom and at.visited==0:
                    name = at.getValueFrom.variable.name
                    var = at.getValueFrom.variable
                    if var.ordered:
                        a = AttrLineOrdered(name, self.bnomogram)
                    else:
                        a = AttrLine(name, self.bnomogram)
                    listOfExcludedValues = []
                    for val in var.values:
                        foundValue = False
                        for same in domain.attributes:
                            if same.visited==0 and same.getValueFrom and same.getValueFrom.variable == var and same.getValueFrom.variable.values[same.getValueFrom.transformer.value]==val:
                                same.setattr("visited",1)
                                a.addAttValue(AttValue(val, mult*cl.beta[same], error = cl.beta_se[same]))
                                foundValue = True
                        if not foundValue:
                            listOfExcludedValues.append(val)
                    if len(listOfExcludedValues) == 1:
                        a.addAttValue(AttValue(listOfExcludedValues[0], 0, error = aprox_prior_error))
                    elif len(listOfExcludedValues) == 2:
                        a.addAttValue(AttValue("("+listOfExcludedValues[0]+","+listOfExcludedValues[1]+")", 0, error = aprox_prior_error))
                    elif len(listOfExcludedValues) > 2:
                        a.addAttValue(AttValue("Other", 0, error = aprox_prior_error))
                    # if there are more than 1 value in the attribute, add it to the nomogram
                    if len(a.attValues)>1:
                        self.bnomogram.addAttribute(a)


                elif at.visited==0:
                    name = at.name
                    var = at
                    a = AttrLineCont(name, self.bnomogram)
                    if self.data:
                        bas = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))
                        maxAtValue = bas[var].max
                        minAtValue = bas[var].min
                    else:
                        maxAtValue = 1.
                        minAtValue = -1.
                    numOfPartitions = 50.
                    d = getDiff((maxAtValue-minAtValue)/numOfPartitions)

                    # get curr_num = starting point for continuous att. sampling
                    curr_num = getStartingPoint(d, minAtValue)
                    rndFac = getRounding(d)

                    while curr_num<maxAtValue+d:
                        if abs(mult*curr_num*cl.beta[at])<aproxZero:
                            a.addAttValue(AttValue("0.0", 0))
                        else:
                            a.addAttValue(AttValue(str(curr_num), mult*curr_num*cl.beta[at]))
                        curr_num += d
                    a.continuous = True
                    at.setattr("visited", 1)
                    # if there are more than 1 value in the attribute, add it to the nomogram
                    if len(a.attValues)>1:
                        self.bnomogram.addAttribute(a)

        self.alignRadio.setDisabled(True)
        self.alignType = 0
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    def lr2Classifier(self, cl):
    
        if len(cl.weights) == 1:
            weights = list(cl.weights[0])
            if self.TargetClassIndex > 0:
                weights = [ -w for w in weights ]
        else:
            weights = list(cl.weights[self.TargetClassIndex])

        domain = cl.domain
        for at in domain.attributes:
            at.setattr("visited",0)

        order = []
        multiple = defaultdict(list)
        for iat,at in enumerate(domain.attributes):
            if at.getValueFrom and hasattr(at.getValueFrom, "variable") \
                and isinstance(at.getValueFrom.variable, Orange.feature.Discrete):
                    var = at.getValueFrom.variable
                    if var not in multiple: 
                        order.append((var,-1))
                    multiple[var].append((at, iat))
            else:
                order.append((at, iat)) #raw variable

        candidateats = []

        for at,iat in order:
            if at not in multiple:
                span = 1.
                average = 0.
                basevar = at
                #check if normalized
                if isinstance(at.getValueFrom, Orange.classification.ClassifierFromVar) \
                    and isinstance(at.getValueFrom.variable, Orange.feature.Continuous) \
                    and isinstance(at.getValueFrom.transformer, Orange.core.NormalizeContinuous):
                    span = at.getValueFrom.transformer.span
                    average = at.getValueFrom.transformer.average
                    basevar = at.getValueFrom.variable
                name = basevar.name

                if self.data:
                    bas = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))
                    maxAtValue = bas[basevar].max
                    minAtValue = bas[basevar].min
                    avgAtValue = bas[basevar].avg
                else:
                    maxAtValue = 1.
                    minAtValue = -1.
                    avgAtValue = 0.

                # get curr_num = starting point for continuous att. sampling

                w = weights[iat]
                betas = [ ((minAtValue-average)/span)*w, ((maxAtValue-average)/span)*w  ]
                rng = max(betas) - min(betas)

                freqbase = ((avgAtValue-average)/span)*w

                candidateats.append((at, name, "cont", minAtValue, maxAtValue, w, average, span, freqbase, rng))

            else:
                found_values = set()
                values = []
                for ati, iati in multiple[at]:
                    val = ati.getValueFrom.variable.values[ati.getValueFrom.transformer.value]
                    values.append((val, weights[iati]))
                    found_values.add(val)
                    
                excluded_values = list(set(at.values) - found_values)
                excluded_name = excluded_values[0] if len(excluded_values) == 1 else "other"
                values.append((excluded_name, 0))

                betas = [ a[1] for a in values ]
                rng = max(betas) - min(betas)

                #the most frequent item has weight 0.
                candidateats.append((at, at.name, "disc", values, 0., rng))

        self.warning(1011)

        biggest_influence = set([a[0] for a in sorted(candidateats, key=lambda x: -x[-1]) ][:SHOW_BIGGEST])

        if len(biggest_influence) < len(candidateats):
            self.warning(1011, "Showing only %d features with highest influence (out of %d)." % (len(biggest_influence), len(candidateats) ))

        hiddenbeta = 0.
        for aaa in candidateats:
            if aaa[0] not in biggest_influence:
                hiddenbeta += aaa[-2]

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue('Constant', weights[-1]+hiddenbeta))
        else:
            self.bnomogram = BasicNomogram(self, AttValue('Constant', weights[-1]+hiddenbeta))

        for aaa in candidateats:

            if aaa[0] in biggest_influence:

                if aaa[2] == "cont":
                    _, name, _, minAtValue, maxAtValue, w, average, span, _, _ = aaa
                    numOfPartitions = 50.
                    d = getDiff((maxAtValue-minAtValue)/numOfPartitions)
                    a = AttrLineCont(name, self.bnomogram)
                    curr_num = getStartingPoint(d, minAtValue)
                    while curr_num<maxAtValue+d:
                        if abs(curr_num*w)<aproxZero:
                            a.addAttValue(AttValue("0.0", 0))
                        else:
                            a.addAttValue(AttValue(str(curr_num), ((curr_num-average)/span)*w))
                        curr_num += d
                    a.continuous = True

                elif aaa[2] == "disc":
                    _, name, _, values, _, _ = aaa
                    a = AttrLine(name, self.bnomogram)
                    for n, val in values:
                        a.addAttValue(AttValue(n, val))
               
                if len(a.attValues)>1:
                    self.bnomogram.addAttribute(a)

        self.alignRadio.setDisabled(True)
        self.alignType = 0
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()


    def svmClassifier(self, cl):

        import orngLinVis

        self.error(0)
        if self.TargetClassIndex == 0 or self.TargetClassIndex == cl.domain.classVar[0]:
            mult = -1
        else:
            mult = 1

        try:
            visualizer = orngLinVis.Visualizer(self.data, cl, buckets=1, dimensions=1)
            beta_from_cl = self.cl.estimator.classifier.classifier.beta[0] - self.cl.estimator.translator.trans[0].disp*self.cl.estimator.translator.trans[0].mult*self.cl.estimator.classifier.classifier.beta[1]
            beta_from_cl = mult*beta_from_cl
        except:
            self.error(0, "orngLinVis.Visualizer error"+ str(sys.exc_info()[0])+":"+str(sys.exc_info()[1]))
#            QMessageBox("orngLinVis.Visualizer error", str(sys.exc_info()[0])+":"+str(sys.exc_info()[1]), QMessageBox.Warning,
#                        QMessageBox.NoButton, QMessageBox.NoButton, QMessageBox.NoButton, self).show()
            return

        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue('Constant', -mult*math.log((1.0/min(max(visualizer.probfunc(0.0),aproxZero),0.9999))-1), 0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue('Constant', -mult*math.log((1.0/min(max(visualizer.probfunc(0.0),aproxZero),0.9999))-1), 0))

        # get maximum and minimum values in visualizer.m
        maxMap = reduce(numpy.maximum, visualizer.m)
        minMap = reduce(numpy.minimum, visualizer.m)

        coeff = 0 #
        at_num = 1
        correction = self.cl.coeff*self.cl.estimator.translator.trans[0].mult*self.cl.estimator.classifier.classifier.beta[1]
        for c in visualizer.coeff_names:
            if type(c[1])==str:
                for i in range(len(c)):
                    if i == 0:
                        if self.data.domain[c[0]].ordered:
                            a = AttrLineOrdered(c[i], self.bnomogram)
                        else:
                            a = AttrLine(c[i], self.bnomogram)
                        at_num = at_num + 1
                    else:
                        if self.data:
                            thickness = float(len(self.data.filter({self.data.domain[c[0]].name:str(c[i])})))/float(len(self.data))
                        a.addAttValue(AttValue(c[i], correction*mult*visualizer.coeffs[coeff], lineWidth=thickness))
                        coeff = coeff + 1
            else:
                a = AttrLineCont(c[0], self.bnomogram)

                # get min and max from Data and transform coeff accordingly
                maxNew=maxMap[coeff]
                minNew=maxMap[coeff]
                if self.data:
                    bas = getCached(self.data, orange.DomainBasicAttrStat, (self.data,))
                    maxNew = bas[c[0]].max
                    minNew = bas[c[0]].min

                # transform SVM betas to betas siutable for nomogram
                if maxNew == minNew:
                    beta = ((maxMap[coeff]-minMap[coeff])/aproxZero)*visualizer.coeffs[coeff]
                else:
                    beta = ((maxMap[coeff]-minMap[coeff])/(maxNew-minNew))*visualizer.coeffs[coeff]
                n = -minNew+minMap[coeff]

                numOfPartitions = 50
                d = getDiff((maxNew-minNew)/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling
                curr_num = getStartingPoint(d, minNew)
                rndFac = getRounding(d)

                while curr_num<maxNew+d:
                    a.addAttValue(AttValue(str(curr_num), correction*(mult*(curr_num-minNew)*beta-minMap[coeff]*visualizer.coeffs[coeff])))
                    curr_num += d

                at_num = at_num + 1
                coeff = coeff + 1
                a.continuous = True

            # if there are more than 1 value in the attribute, add it to the nomogram
            if len(a.attValues)>1:
                self.bnomogram.addAttribute(a)
        self.cl.domain = orange.Domain(self.data.domain.classVar)
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    # Input channel: the rule classifier (from CN2-EVC only)
    def ruleClassifier(self, cl):
        def selectSign(oper):
            if oper == orange.ValueFilter_continuous.Less:
                return "<"
            elif oper == orange.ValueFilter_continuous.LessEqual:
                return "<="
            elif oper == orange.ValueFilter_continuous.Greater:
                return ">"
            elif oper == orange.ValueFilter_continuous.GreaterEqual:
                return ">="
            else: return "="

        def getConditions(rule):
            conds = rule.filter.conditions
            domain = rule.filter.domain
            ret = []
            if len(conds)==0:
                ret = ret + ["TRUE"]
            for i,c in enumerate(conds):
                if i > 0:
                    ret[-1] += " & "
                if type(c) == orange.ValueFilter_discrete:
                    ret += [domain[c.position].name + "=" + str(domain[c.position].values[int(c.values[0])])]
                elif type(c) == orange.ValueFilter_continuous:
                    ret += [domain[c.position].name + selectSign(c.oper) + "%.3f"%c.ref]
            return ret

        self.error(1)
        if not len(self.data.domain.classVar.values) == 2:
            self.error(1, "Rules require binary classes")
        classVal = cl.domain.classVar
        att = cl.domain.attributes

        if self.TargetClassIndex == 0 or self.TargetClassIndex == cl.domain.classVar[0]:
            mult = 1.
        else:
            mult = -1.

        # calculate prior probability (from self.TargetClassIndex)
        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue("Constant", 0.0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue("Constant", 0.0))
        self.cl.setattr("rulesOrdering", [])
        for r_i,r in enumerate(cl.rules):
            a = AttrLine(getConditions(r), self.bnomogram)
            self.cl.rulesOrdering.append(getConditions(r))
            if r.classifier.defaultVal == 0:
                sign = mult
            else: sign = -mult
            a.addAttValue(AttValue("yes", sign*cl.ruleBetas[r_i], lineWidth=0, error = 0.0))
            a.addAttValue(AttValue("no", 0.0, lineWidth=0, error = 0.0))
            self.bnomogram.addAttribute(a)

        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()

    def generalClassifier(self, cl):
        if not self.learn_data:
            print('No data :(')
            return 1
        else:
            self.data = self.learn_data
        
        self.stat = orange.DomainBasicAttrStat(self.data,)
        
        if self.bnomogram:
            self.bnomogram.destroy_and_init(self, AttValue("Constant", 0.0))
        else:
            self.bnomogram = BasicNomogram(self, AttValue("Constant", 0.0))

        self.generatedData = self.generateData(cl)
        #self.data = cl.data
        classVal = cl.domain.classVar
        targetClass = classVal[self.TargetClassIndex].value
        
        # Calculate distributions
        #new_dist = Orange.statistics.contingency.Domain(self.data)
        cl.conditionalDistributions = Orange.statistics.contingency.Domain(self.generatedData)
        cl.conditionalDistributions.normalize()
        cl.distribution = Orange.statistics.distribution.Distribution(classVal.name, self.generatedData)
        cl.distribution.normalize()
        #self.calculateDistributions(cl)                 

        # calculate prior probability
        dist1 = max(aproxZero, 1-cl.distribution[classVal[self.TargetClassIndex]])
        dist0 = max(aproxZero, cl.distribution[classVal[self.TargetClassIndex]])
        prior = dist0/dist1
        
        
        for at, attribute  in enumerate(cl.domain.attributes):
            a = None                        
            if attribute.varType == orange.VarTypes.Discrete:
                if attribute.ordered:
                    a = AttrLineOrdered(attribute.name, self.bnomogram)
                else:
                    a = AttrLine(attribute.name, self.bnomogram)
                for cd in cl.conditionalDistributions[at].keys():
                    conditionalDistribution = cl.conditionalDistributions[at][cd]
                    
                    targetClassDistribution = conditionalDistribution[classVal[self.TargetClassIndex]]
                    conditional0 = max(targetClassDistribution, aproxZero)
                    conditional1 = max(1-targetClassDistribution, aproxZero)
                    beta = math.log(conditional0/conditional1/prior)
                    

                    a.addAttValue(AttValue(cd, beta))

            else:
                a = AttrLineCont(attribute.name, self.bnomogram)
                numOfPartitions = 50

                if self.data:
                    maxAtValue = self.stat[at].max
                    minAtValue = self.stat[at].min
                else:
                    maxAtValue = cl.conditionalDistributions[at].keys()[len(cl.conditionalDistributions[at].keys())-1]
                    minAtValue = cl.conditionalDistributions[at].keys()[0]

                d = maxAtValue-minAtValue
                d = getDiff(d/numOfPartitions)

                # get curr_num = starting point for continuous att. sampling
                curr_num = getStartingPoint(d, minAtValue)
                rndFac = getRounding(d)

                values = []
                for cn in numpy.arange(minAtValue, maxAtValue, d):
                #for i in range(2*numOfPartitions):
                    if cn>=minAtValue and cn<=maxAtValue:
                        
                        d_filter = filter(lambda x: x>cn-d/2 and x<cn+d/2, cl.conditionalDistributions[at].keys())
                        if len(d_filter)>0:
                            cd = cl.conditionalDistributions[at]
                            conditional0 = avg([cd[f][classVal[self.TargetClassIndex]] for f in d_filter])
                            conditional0 = min(1-aproxZero,max(aproxZero,conditional0))
                            conditional1 = 1-conditional0
                            try:
                                # compute error of loess in logistic space
                                var = avg([cd[f].variances[self.TargetClassIndex] for f in d_filter])
                                standard_error= math.sqrt(var)
                                rightError0 = (conditional0+standard_error)/max(conditional1-standard_error, aproxZero)
                                leftError0  =  max(conditional0-standard_error, aproxZero)/(conditional1+standard_error)
                                se = (math.log(rightError0) - math.log(leftError0))/2
                                se = math.sqrt(math.pow(se,2)+math.pow(priorError,2))

                                # add value to set of values
                                a.addAttValue(AttValue(str(round(cn,rndFac)),
                                                       math.log(conditional0/conditional1/prior),
                                                       lineWidth=thickness,
                                                       error = se))
                            except:
                                pass
                a.continuous = True

        
            if a and len(a.attValues)>1:
                self.bnomogram.addAttribute(a)

        self.alignRadio.setDisabled(False)
        self.graph.setScene(self.bnomogram)
        self.bnomogram.show()
    
    def calculateDistributions(self, cl):
        #condition_data = Orange.data.Table([d for d in self.data if d[attribute] == value])
        class_name = cl.domain.classVar.name
        #cd_count = len(condition_data)
        #class_count = [1 for d in condition_data if d[class_name] == 1]
        
        d_count = len(self.data)
        count = {}
        ccount = {}
        
        for instance in self.data:
            instance_cls = str(instance[class_name])
            ccount[instance_cls] = ccount.get(instance_cls, 0) + 1
            for att_index, att_value in enumerate(instance):
                att_value = str(att_value)
                count[att_index] = count.get(att_index, {})                 
                count[att_index][att_value] = count[att_index].get(att_value, {})                 
                count[att_index][att_value][instance_cls] = count[att_index][att_value].get(instance_cls, 0) + 1
            
        distributions = {}    
        for att_index, att in enumerate(cl.domain.attributes):
            distributions[att.name] = {}
            for value in att.values:
                distributions[att.name][value] = {}
                for c in cl.domain.classVar.values:
                    num = count[att_index][value].get(c, 0)
                    distributions[att.name][value][c] = num*1.0 / ccount[c]
        #cl.conditionalDistributions = distributions
        dist = Orange.statistics.contingency.Domain(self.data)
        cl.conditionalDistributions = Orange.statistics.contingency.Domain(self.data)
        
        general_dist = {}
        for c in cl.domain.classVar.values:
            general_dist[c] = ccount[c] / d_count
        
        #cl.distribution = general_dist
        cl.distribution = Orange.statistics.distribution.Distribution(self.data)
        
        return distributions
        
    def generateData(self, cl):
        combinations = []
        import numpy as np
        for attr in cl.domain.attributes:
            if attr.varType == orange.VarTypes.Discrete:
                values = attr.values
                combinations.append(values)
            else:
                attr_stat = self.stat[attr.name]
                start = attr_stat.min
                stop = attr_stat.max
                step = (stop - start)*1.0/3
                values = [n for n in np.arange(start, stop, step)]
               # values.append(start)
                values.append(stop)
               # values.sort()
                combinations.append(values)
        
        #data.domain.attributes[0].values
        #Orange.data.Table
        #new_data = data.clone() 
        
        # I think i shoud specify continious type somewhere
        
        new_data = Orange.data.Table(cl.domain, [])
        for combination in itertools.product(*combinations):
            combination_list = list(combination)
            combination_list.append('?')
            di = Orange.data.Instance(cl.domain, combination_list)
            c = cl(di)
            combination_list.pop()
            combination_list.append(c)
            new_data.append(combination_list)
        
        return new_data
        
    def generalClassifierBayes(self, cl):
        
       # classVal = cl.domain.classVar
       # att = cl.domain.attributes
        
        domain = cl.domain

        # New sample data
        combinations = []
        combinations_count = 1;
        for attr in domain.attributes:
            values = attr.values
            combinations.append(values)
            combinations_count = combinations_count * len(values)
        
        import itertools
        
        #data.domain.attributes[0].values
        #Orange.data.Table
        #new_data = data.clone() 
        new_data = Orange.data.Table(domain, [])
        for combination in itertools.product(*combinations):
            combination_list = list(combination)
            combination_list.append('?')
            di = Orange.data.Instance(domain, combination_list)
            c = cl(di)
            combination_list.pop()
            combination_list.append(c)
            new_data.append(combination_list)
            
        self.bayes = orange.BayesLearner(new_data)
        self.nbClassifier(self.bayes)


    def initClassValues(self, classValue):
        self.targetCombo.clear()
        self.targetCombo.addItems([str(v) for v in classValue])

    def setData(self, orange_data):
        self.myvar = 'Gregas var :D'
        self.learn_data = orange_data

    def classifier(self, cl):
        self.closeContext()
        self.error(2) 

        oldcl = self.cl
        self.cl = None
        
        if cl:
            self.cl = cl
                 
        if not oldcl or not self.cl or not oldcl.domain == self.cl.domain:
            if self.cl:
                self.initClassValues(self.cl.domain.classVar)
            self.TargetClassIndex = 0
            
        self.data = getattr(self.cl, "data", None)

        if self.data and self.data.domain and not self.data.domain.classVar:
            self.error(2, "Classless domain")
            # Here it said "return", but let us report an error and clean up the widget
            self.cl = self.data = None

        self.openContext("", self.data)
        if not self.data:
            self.histogramCheck.setChecked(False)
            self.histogramCheck.setDisabled(True)
            self.histogramLabel.setDisabled(True)
            self.CICheck.setChecked(False)
            self.CICheck.setDisabled(True)
            self.CILabel.setDisabled(True)
        else:
            self.histogramCheck.setEnabled(True)
            self.histogramCheck.makeConsistent()
            self.CICheck.setEnabled(True)
            self.CICheck.makeConsistent()
        self.updateNomogram()

    def setTarget(self):
        self.updateNomogram()

    def updateNomogram(self):
##        import orngSVM

        def setNone():
            for view in [self.footer, self.header, self.graph]:
                scene = view.scene()
                if scene:
                    for item in scene.items():
                        scene.removeItem(item)

        if self.data and self.cl: # and not type(self.cl) == orngLR_Jakulin.MarginMetaClassifier:
            #check domains
            for at in self.cl.domain:
                if at.getValueFrom and hasattr(at.getValueFrom, "variable"):
                    if (not at.getValueFrom.variable in self.data.domain) and (not at in self.data.domain):
                        return
                else:
                    if not at in self.data.domain:
                        return

        if isinstance(self.cl, orange.BayesClassifier):
            self.nbClassifier(self.cl)
        elif isinstance(self.cl, orange.RuleClassifier_logit):
            self.ruleClassifier(self.cl)

        elif isinstance(self.cl, orange.LogRegClassifier):
            # get if there are any continuous attributes in data -> then we need data to compute margins
            cont = False
            if self.cl.continuizedDomain:
                for at in self.cl.continuizedDomain.attributes:
                    if not at.getValueFrom:
                        cont = True
            if self.data or not cont:
                self.lrClassifier(self.cl)
            else:
                setNone()
        elif isinstance(self.cl, Orange.classification.svm.LinearClassifier):
            self.lr2Classifier(self.cl)
        else:
            
            self.generalClassifier(self.cl)
        #self.generalClassifier(self.cl)
            
        if self.sort_type>0:
            self.sortNomogram()

    def sortNomogram(self):
        def sign(x):
            if x<0:
                return -1;
            return 1;
        def compare_to_ordering_in_rules(x,y):
            return self.cl.rulesOrdering.index(x.name) - self.cl.rulesOrdering.index(y.name)
        def compare_to_ordering_in_data(x,y):
            return self.data.domain.attributes.index(self.data.domain[x.name]) - self.data.domain.attributes.index(self.data.domain[y.name])
        def compare_to_ordering_in_domain(x,y):
            return self.cl.domain.attributes.index(self.cl.domain[x.name]) - self.cl.domain.attributes.index(self.cl.domain[y.name])
        def compate_beta_difference(x,y):
            return -sign(x.maxValue-x.minValue-y.maxValue+y.minValue)
        def compare_beta_positive(x, y):
            return -sign(x.maxValue-y.maxValue)
        def compare_beta_negative(x, y):
            return sign(x.minValue-y.minValue)

        if not self.bnomogram:
            return
        if self.sort_type == 0 and hasattr(self.cl, "rulesOrdering"):
            self.bnomogram.attributes.sort(compare_to_ordering_in_rules)
        elif self.sort_type == 0 and self.data:
            self.bnomogram.attributes.sort(compare_to_ordering_in_data)
        elif self.sort_type == 0 and self.cl and self.cl.domain:
            self.bnomogram.attributes.sort(compare_to_ordering_in_domain)
        if self.sort_type == 1:
            self.bnomogram.attributes.sort(compate_beta_difference)
        elif self.sort_type == 2:
            self.bnomogram.attributes.sort(compare_beta_positive)
        elif self.sort_type == 3:
            self.bnomogram.attributes.sort(compare_beta_negative)

        # update nomogram
        self.showNomogram()


    def setProbability(self):
        if self.probability and self.bnomogram:
            self.bnomogram.showAllMarkers()
        elif self.bnomogram:
            self.bnomogram.hideAllMarkers()

    def setBaseLine(self):
        if self.bnomogram:
            self.bnomogram.showBaseLine(True)

    def menuItemPrinter(self):
        canvases = header, graph, footer = self.header.scene(), self.graph.scene(), self.footer.scene()
        # all scenes together
        scene_confed = QGraphicsScene(0, 0, max(c.width() for c in canvases), sum(c.height() for c in canvases))
        # add items from header
        header_its = header.items()
        for it in header_its:
            scene_confed.addItem(it)
        # add items from graph
        graph_its = graph.items()
        for it in graph_its:
            scene_confed.addItem(it)
            it.moveBy(0., header.height())
        # add from footer
        footer_its = footer.items()
        for it in footer_its:
            scene_confed.addItem(it)
            it.moveBy(0.,header.height() + graph.height())
        try:
            import OWDlgs
        except:
            print "Missing file 'OWDlgs.py'. This file should be in OrangeWidgets folder. Unable to print/save image."
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(scene_confed, parent=self)
        sizeDlg.exec_()

        # set all items back to original canvases            
        for it in header_its:
            header.addItem(it)
        for it in graph_its:
            graph.addItem(it)
            it.moveBy(0., -header.height())
        for it in footer_its:
            footer.addItem(it)
            it.moveBy(0, - header.height() - graph.height())
        self.showNomogram()

    # Callbacks
    def showNomogram(self):
        if self.bnomogram and self.cl:
            #self.bnomogram.hide()
            self.bnomogram.show()
            self.bnomogram.update()


# test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNomogramGeneral()
    ow.show()

    #data = orange.ExampleTable("C:\Python27\Lib\site-packages\Orange\datasets\heart_disease.tab")
    debug_data = orange.ExampleTable('C:\Python27\Lib\site-packages\Orange\datasets\\titanic.tab')
    debug_data = orange.ExampleTable('C:\Python27\Lib\site-packages\Orange\datasets\\iris.tab')
    ow.setData(debug_data)

    import orngTree
    debug_tree = orngTree.TreeLearner(debug_data)
    debug_rforest = Orange.ensemble.forest.RandomForestLearner(debug_data)
    debug_major = orange.MajorityLearner(debug_data)

    debug_bayes = orange.BayesLearner(debug_data)
    debug_bayes.setattr("data", debug_data)
    
    ow.classifier(debug_tree)
    
    #rules = orange.RuleLearner_logit(data)

    # here you can test setting some stuff

    a.exec_()

    # save settings
    ow.saveSettings()

