---
layout: default
Title: Search
permalink: /search
---

Search powered by lunr.js
<form action="/search" id="site_search">
                <input type="text" id="search_box" name="query">
                <input type="submit" value="Search">
              </form> 
              
### Site Search Results

<ul id="search_results">Search Loading</ul>

### Documentation Search Results

<ul id="doc_results"></ul>

<script src="/js/lunr.min.js"></script>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>
<script src="/js/search.js"></script>
<script> 
window.onload = function() {
	var param = location.search;
	qstart = param.search("query=");
	qend = qstart + "query=".length;
	var res = param.slice(qend);
	$("#search_box").val(res);
	$("#site_search").trigger("submit");
};

</script>


