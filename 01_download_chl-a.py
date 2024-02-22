import requests

base_url = 'https://zenodo.org/records/7092220/files/Y{year}{month:02d}.L3m_MO_CHL_chlor_a_4km.nc?download=1'

for year in range(1998, 2021):
    for month in range(1, 13):
        url = base_url.format(year=year, month=month)
        filename = './01_chl-a_new_data/Y{year}{month:02d}.L3m_MO_CHL_chlor_a_4km.nc'.format(year=year, month=month)

        response = requests.get(url)

        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f'文件 {filename} 下载成功！')
        else:
            print(f'无法下载文件 {filename}。')