from surfquakecore.utils.manage_catalog import WriteCatalog

catalog_path = "/Volumes/LaCie/all_andorra/catalog/catalog_obj.pkl"
output_path = "/Volumes/LaCie/all_andorra/catalog/catalog_surf.txt"
wc = WriteCatalog(catalog_path)
print(wc.show_help())
help(wc.filter_time_catalog)
help(wc.filter_geographic_catalog)
#catalog_filtered = wc.filter_time_catalog(starttime="30/01/2022, 00:00:00.0", endtime="20/02/2022, 00:00:00.0")

#catalog_filtered = wc.filter_geographic_catalog(catalog_filtered, lat_min=42.1, lat_max=43.0, lon_min=0.8, lon_max=1.5,
#                                                depth_min=-10, depth_max=20, mag_min=3.4, mag_max=3.9)

wc.write_catalog_surf(catalog=None, output_path=output_path)
#wc.write_catalog_surf(catalog=catalog_filtered, output_path=output_path)

# Now you can also save the filtered catalog
#catalog_filtered.plot(projection='local')