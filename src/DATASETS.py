LDA_DATASETS = [
    "max(all)+boilerplate*10", 
    "max(all)+meta-description*10", 
    "max(h1, title)",
    "max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description * 10 + meta-keywords * 10 + max(boilerplate, boilerpipe) * 5", 
    "body+title*10", 
]

DATASETS = [
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description * 10 + meta-keywords * 10 + max(boilerplate, boilerpipe) * 5',
    'max(all)+boilerplate*10',
    'max(all)+meta-description*10',
    'max(h1, title)',
    'max(boilerpipe, boilerplate)', 
    'body+title*10',

    'title',
    'h1',
    'body',
    'other',
    'meta-description',
    'meta-keywords',
    'h2',
    'h3',
    'img',
    'a',
    'boilerplate',
    'boilerpipe',
    'max(all)',

    'max(body, boilerpipe, boilerplate)', 
    'max(body, boilerpipe, boilerplate, meta-description)', 
    'max(body, boilerpipe, boilerplate, meta-description, meta-keywords)', 

    'body+h1*10',
    'body+body*10',
    'body+other*10',
    'body+meta-description*10',
    'body+boilerplate*10',
    'body+boilerpipe*10',

    'body+h1*5',
    'body+title*5',
    'body+body*5',
    'body+other*5',
    'body+meta-description*5',
    'body+boilerplate*5',
    'body+boilerpipe*5',

    'max(all)+h1*5',
    'max(all)+title*5',
    'max(all)+body*5',
    'max(all)+other*5',
    'max(all)+meta-description*5',
    'max(all)+boilerplate*5',
    'max(all)+boilerpipe*5',

    'max(all)+h1*10',
    'max(all)+title*10',
    'max(all)+body*10',
    'max(all)+other*10',

    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords + img + a + body + boilerplate + max(boilerplate, boilerpipe)',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords + img + a + body + boilerplate + max(boilerplate, boilerpipe) * 5',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords + img + a + body + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords*5 + img + a + body + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords*10 + img + a + body + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords*10 + a + body + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords*10 + body + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description + meta-keywords*10 + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description * 10 + meta-keywords + boilerplate + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description * 10 + meta-keywords * 10 + boilerplate * 0.5 + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + max(title, h1, h2) + max(title, h1, h2, h3) + meta-description * 10 + meta-keywords * 10 + max(boilerplate, boilerpipe) * 10',
    'max(title, h1) * 10 + meta-description * 10 + meta-keywords * 10 + max(boilerplate, boilerpipe) * 10',
]

