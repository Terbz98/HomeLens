# Three-minute project proposal script

About 500 words, which is a little over three minutes at a calm speaking pace. Every figure is copied from
[evaluation.md](evaluation.md).

---

**[0:00] The idea**

When you look at a house on Google Maps, you can see it, but you cannot tell what it costs. HomeLens
is a Chrome extension that answers that. You click a house, confirm the address, and a side panel
shows four numbers: the price to buy it and the monthly rent, each with and without furniture. You
do not type in bedrooms or floor area. For Seattle the app looks them up in the county's public
record of the house. We also tested whether a street photo of the outside makes the estimate better.

**[0:35] The data**

We support detached single-family homes in three cities: Seattle, San Francisco and Portland. For
Seattle we use real public sale records from King County: 9,652 house sales from the last three
years. For all three cities we use Zillow Research data, which gives the typical home value and rent
for each ZIP code. We did not scrape any website. One important limit: Google's terms do not allow
feeding Street View images to a model, so our photos come from Mapillary, which is openly licensed.

**[1:10] The models**

We compare four levels. First a simple baseline, such as the city median. Second location only:
nearest neighbors on latitude and longitude. Third location plus the county's public record of the
house, such as living area, building grade and age, in a gradient boosted tree model. Fourth, a
street photo on top: a pretrained image network, or the Harris corner detector from our DSP lab,
turns the photo into numbers that correct the estimate. We train on older sales and test on the
newest ones, and no house appears on both sides.

**[1:45] Results**

On the newest 1,931 Seattle sales, which the models never saw, the city median is off by $517,088
on average and location only by $363,904. With the house record, on almost the same 1,896 sales, the
error drops to $163,388, a typical miss of 9.2 percent, so neighbouring houses finally get different
prices. Do photos add anything? On top
of location only, a small but real gain: from $403,273 to $400,472. On top of the house record, no
measurable difference. Recent photos did not help more than old ones.

**[2:20] Being honest about the numbers**

Three things we show instead of hide. One: a photo cannot see furniture, so the furnished prices add
a fixed, documented furniture package, labeled as an assumption. Two: for San Francisco and Portland
we only have ZIP-level data, so those prices are labeled area-level, not house-level. Three: we only
show a price range where we could check it. For Seattle, the range was tested on held-out sales and
its measured coverage is shown next to it.

**[2:50] Close**

HomeLens runs today: the extension, the backend, the trained models and the tests. The next step
would be photos that face the house and an image network tuned on houses. Thank you.
