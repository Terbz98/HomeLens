# Three-minute project proposal script

About 470 words, which is a little over three minutes at a calm speaking pace. Every figure is copied from
[evaluation.md](evaluation.md).

---

**[0:00] The idea**

When you look at a house on Google Maps, you can see it, but you cannot tell what it costs. HomeLens
is a Chrome extension that answers that. You click a house, confirm the address, and a side panel
shows four numbers: the price to buy it and the monthly rent, each with and without furniture. You
do not type in bedrooms or floor area. The app only needs to know where the house is. We also tested
whether a street photo of the outside makes the estimate better.

**[0:35] The data**

We support detached single-family homes in three cities: Seattle, San Francisco and Portland. For
Seattle we use real public sale records from King County: 9,652 house sales from the last three
years. For all three cities we use Zillow Research data, which gives the typical home value and rent
for each ZIP code. We did not scrape any website. One important limit: Google's terms do not allow
feeding Street View images to a model, so our photos come from Mapillary, which is openly licensed.

**[1:10] The models**

We compare three levels. First a simple baseline, such as the city median. Second a location-only
model: nearest neighbors on latitude and longitude. Third a location-plus-photo model, where a
pretrained network turns the photo into numbers and a regression uses them to correct the location
estimate. To avoid leakage, we train on older sales and test on the newest ones, and no house
appears on both sides.

**[1:45] Results**

On 1,931 Seattle sales the model never saw, the city-median baseline is off by $517,088 on average.
The location-only model is off by $363,904. The median price is $1,125,000, so location alone is
still wrong by about a third. Does the photo close that gap? Mapillary had a usable photo for 57
percent of the houses we sampled. On the 327 test sales with a photo, location only is off by
$403,927 and location plus photo by $403,728. That is $200 better, which is far inside the noise.
So our measured answer is no: with dashcam street photos and a general-purpose image network, the
photo did not help, and the app keeps using location only.

**[2:20] Being honest about the numbers**

Three things we show instead of hide. One: a photo cannot see furniture, so the furnished prices add
a fixed, documented furniture package, labeled as an assumption. Two: for San Francisco and Portland
we only have ZIP-level data, so those prices are labeled area-level, not house-level. Three: we only
show a price range where we could check it. For Seattle, the range was tested on held-out sales and
its measured coverage is shown next to it.

**[2:50] Close**

HomeLens runs today: the extension, the backend, the trained models and the tests. The next step
would be photos that face the house and a network tuned on houses. Thank you.
