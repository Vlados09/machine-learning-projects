import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting

    # Method performing retrieval for specified query
    def forQuery(self, query):
        
        relevant_docs = set() #Stores numbers of the relevant documents
       
        # Extract only relevant documents. i.e: those that have common terms 
        # with a query.
        for term in query:
            if term in self.index.keys():
                relevant_docs.update(((self.index).get(term)).keys())
        
        # Stores the sum of qi*di for each document
        similarity = dict.fromkeys(relevant_docs, -1) 
        # Stores the sum of squared document size.
        doc_size = dict.fromkeys(relevant_docs, 0) 
        
        for term in self.index:
            value = self.index[term] #Access the documents that have this term.
            for doc_num in value:
                if doc_num in relevant_docs:
                    count = value[doc_num] #Access the number of this term in the document.
                    doc_size[doc_num] += count*count
                    if term in query:
                        #Choose which term weighting schema to use.
                        if (self.termWeighting == 'binary'):
                            similarity[doc_num] += 1
                        elif (self.termWeighting == 'tf'):
                            similarity[doc_num] += (value[doc_num] * query[term])
                        else:
                            idf = math.log10(len(relevant_docs)/len(value))
                            similarity[doc_num] += (idf * (value[doc_num] * query[term]))
        
        # Sotres the final result - size adjusted similarity measure.
        results = dict.fromkeys(relevant_docs, 0)
        
        # Calculate result for each relevant document.
        for doc in relevant_docs:
            results[doc] = similarity[doc]/(math.sqrt(doc_size[doc]))
        
        # Sort the results to have highest scoring at the beggining
        sorted_results = sorted(results, key=results.get, reverse = True)
                      
        return sorted_results[:10] #Return top 10 results.