from transformers import pipeline
import torch
import librosa
import numpy as np
from pydub import AudioSegment

# Function to extract a segment of audio from an audio file
def extract_audio_segment(audio_file_path, sample_rate, start_time, end_time):
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_file_path, sr=sample_rate, offset=start_time, duration=end_time-start_time)
    return audio

def process_pauses(pauses, audio_file_path, sample_rate = 16000):
    results = []
  
    # Define the path to your fine-tuned model and audio file
    model_path = "/home/itfelicia/clap_finetune_model/model_1"
    # Load the fine-tuned model using the pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_classifier = pipeline(task="zero-shot-audio-classification", model=model_path, device=device)
    
    # Define candidate labels (example)
    
    candidate_labels = ["Silence","Vine Boom", "Laughter", "Sad Trombone", "Air Horn", "Explosion", "Applause", "Clapping", "Whistle", "Record Scratch", "Birds Chirping", "Typing Sound", "Camera Shutter/Click", "Heartbeat", "Phone Notification (Ding or Buzz)", "Swoosh or Whoosh", "Sliding Door or Doorbell", "Cartoon Boing", "Laugh Track", "Tada Sound", "Thunder", "Ding (or Bell)", "Drumroll", "Rimsh", "Crickets", "Zoom Sound", "Water","Splash", "Coughing", "Sneeze","Chime", "Coin Drop", "Wind", "Dog Barking", "Cow mooing","Cat meowing", "Music Stinger", "Siren" , 'LAKAD_MATATAG_NORMALIN_NORMALI', 'ping', 'FIRE_IN_THE_HOLE_Geometry_Dash', 'EA_Games', 'Poi', 'surprise_mother_fuer', 'Transformers_transforming', 'pop', 'Chipi_chipi_chapa_chapa', 'let_him_cook_now', 'reaverkill5', 'Falcon_Punch', 'Subway_mcfrappe_skibidi_mbappe', 'Hellou', 'bomb_has_been_planted', 'Error_SOUNDSS', 'Uruha_Rushia_-_NEEEEEE_v2', 'This_is_Sparta', 'Despicable_me_whistle_song', 'Joseph_Joestar-OH_SHIT', 'Valorant_teleporter', 'Shut_up', 'Ramiel_Attack_Scream', 'come_on', 'tear_paper', 'Crowd_Shocked', 'Mario_Wins', 'GBF_AMAZING', 'Gambit_oooimabotamakanameformyselfheah', 'Palpatine_Do_It', 'Huh_Cat', 'China_airlines', 'Bye_have_a_great_time_Twitch_sound', 'I_WOKE_UP_IN_A_NEW_BUGATTI', 'you_promised_my_son_free_robux', 'skedaddle', 'yippee_tbh', 'oh_shit_-ohshitohshit_...oh...shit...', 'Sicko_Mode_Meme_SFX', 'CLARISSE_GANBARE_DANCHOU', 'Single_Swish_01', 'Jeopardy_Correct_Answer', 'Jurassic_Park', 'Meme_omgs', 'FBI_OPEN_UP_with_explosion', 'Minecraft_Lava', 'The_bluetooth_device_is_ready_to_pair', 'Coin_Mario', 'Metallic_Clank', 'HORROR_SCARE', 'Pluh', 'TU_TU_TU_DU_MAX_VERSTAPPEN', 'Oggy_loud', 'Pekora_HOREE_SHIIT', 'WE_ARE_THE_CHAMPIONS', 'WIDE_PUTIN_MEME', 'Indian_Scammer', 'Crickets', 'STICKING_OUT_UR_GYAT_FOR_THE_RIZZLER', 'meow_meow_meow_tiktok', 'nuclear_launch_detected', 'Wet_Slow_Fart', 'Over_9000', 'Amongus_Sus', 'Very_nice_Caesar-chan', 'rust_fake_footsteps', 'English_or_spanish', 'Drumroll', 'Gojo_domain_expansion', 'Pillar_Men_Awaken', 'DO_NOT_TOUCH', 'Nuclear_Fart', 'Shirakami_Fubuki_-_Glasses_are_really_versatile', 'Sugoi_Sugoi', 'SCOTLAND_FOREVER', 'bass_boost', 'anime_ahh', 'Lava_death', 'slap31', 'Ya_no_aguanto_ms', 'miguel_o_harris_spider-man_2099', 'Oh_No_No_No_Tik_Tok_Song_Sound_Effect', 'fnaf_2_ambience_1', 'AYAYA_AYAYA', 'SpongeBob_Fail', 'Chewbacca', 'sparklessss', 'AAAAAaaah_GRILL', 'heheheha', 'thick_of_it_sus', 'Fnaf_2_Hallway', 'Disappear', 'Mii_Channel_Music', 'correct_answer_sound', 'Weezer_Riff', 'pretty_cool_bananas', 'I_am_Batman', 'You_just_have_to_say_that_youre_fine', 'YEE-HAW', 'Car_Crash_SFX', 'Jotaros_Yare_Yare_Daze', 'BELLIGOL_BELLIGOL_BELLIGHAM', 'YOUR_PHONE_RINGING', 'BRUH_sound_effect', 'Kamen_Rider_Build_BEST_MATCH', 'bing_chilling', 'Ali-A_Intro', 'Nanachi_crisp', 'RONALDO_SIUUUU', 'Oh_Hell_No_Vine', 'U_Got_That_meme', 'I_am_going_to_commit_great_crime', 'AMONGUS', 'Pikachu_Thunderbolt', 'Enemy_Spotted', 'ara_ara_ma_ma', 'suspense_build_up', 'Typing_Sound_efffect', 'I_am_Steve', 'FART_WET_DONTEFLON', 'mariokart_toad_shouting_ree', 'You_Need_to_Leave', 'clash_royale_laugh', 'I_like_ya_cut_G', 'rizzler', 'lol_mising_ping', 'Tuco_GET_OUT', 'LIBERTYLIBERTYLIBERTY', 'Haram_PEPPA_PIG_NI-', 'Valorant_Kill_Sound', 'What_The_Hell_Meme_Sound_Effect', 'phasmophobia_-_ghost_attack_1', 'Last_of_Us_Clicker_sound', 'Valorant_defuse_spike', 'static_noise', 'Meme_final', 'Lactic_acid_pain_OG', 'heavenly_musiic', 'Explosion_meme', 'Facebook_Messenger', 'yoda_screaming', 'Lets_Go_go_go', '-Click-_Nice', 'TF2_bonk', 'Hey_thats_pretty_good', 'Snoop_Dogg_meme', 'Engine_Rev', 'Headshot_cs_go_helmet_sound', 'Naruto_Sad_Song', 'jixaw_metal_pipe_falling_sound', 'Roblox_Explosion_Sound', 'MOOSCLES_ARE_GETTING_BIGGER', 'Yakety_Sax', 'ALL_MY_FELLAS', 'Mouse_Click', 'WOW_wink', 'Roblox_Death', 'Wrong', 'Thick_of_it', 'looksmaxxing', 'Fart', 'Minecraft_Drop_Item_Block_Sound_Effect', 'gordon_ramsay_dramatic_sound', 'Drama', 'Cloaker_Payday_2', 'AWP_CSGO', 'Bad_Piggiess', 'Cristiano_Ronaldo', 'mmm_so_good_and_tasty', 'Discord_Notification', 'Sirxrix', 'Pop_SFX', 'Trap', 'horse_wheezing', 'nelly_ahh', 'ksi_new', 'The_Undertaker_Bell', 'Suiiiiiiiiiii', 'Clash_of_Clans_Startup', 'Record_Scratch_WUT', 'aizen_yokoso', 'Classic_Pokemon_Heal', 'Sponge_Stank_Noise', 'Wait_wait_wait_what_the_hell_legend_sound', 'America_Ya_Hallo', 'Suspense.dundunduun', 'U_Cant_Touch_This', 'sword', 'Holy_Sound', 'Lagging_loading', 'INDIAN_SONG_FULL', 'Hello_guys', 'Ginagawa_mo', 'SPONGEBOB_A_FEW_MOMENTS_LATER', 'HAha_funny_laugh', 'Wilhelm_Scream', 'NANI_SOREEEE', 'French_meme_song', 'instagram_thud', 'F1', '8_bit_foot_steps', 'CaseOh_at_burger_king', 'screech_noise_doors_roblox', 'Shing', 'Hello_your_computer_has_virus', 'Tyler1_Machinegun', 'Pouring_Rain', 'I_need_healing.', 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'Rimshot', 'yap_session_loud', 'EZ4ENCE', 'Cartoon_Slip_and_Fall', 'Dj_scratch_hold_up', 'Tuturu', 'skull_trumpet', 'Fat', 'Happy_happy_happy_cat', 'Windows_XP_Error', 'What_Bottom_Text_Meme_Sanctuary_Guardian_-_S', 'Gegagedigedagedago_Full', 'NFL_Bass_Boosted', 'OH_HELLO_THERE.', 'The_Price_is_Right_Losing_Horn', 'INDIAN_EARRAPE_BY_ANTON', 'tinder_sound', 'tom_da_tank_meme', 'CaseOh_Burger_King_song', 'Among_us', 'baby_laughing_meme', 'totally_not_a_suspicious_button', 'lion_scan', 'Mouse_Click_Sound', 'BASS_BOOST_discord_call', 'Ay_Ay_Ay_Im_your_little_butterfly', 'Snake_death_scream', 'Got_a_Item_BOTW', 'Donald_Trump_I_am_the_chosen_one', 'Youre_Not_Really_Fine', 'Running_footsteps', 'iPhone_Notification', 'One_Eternity_Later', 'Minato_Aqua_-_Otsuaqua', 'Why_were_wtill_here', 'Thunder', 'quack', 'The_Nut_Button', 'They_know_me_as_the_rizzler', 'Vengaboys_Boom_Boom_Boom_Boom', 'White_tee_RIZZ', 'Angry_Indian_Scammer', 'DEJA_VU_MEME', 'MSN_Nudge', 'AMARILLO_x4_LOS_PLATANOS', 'Ganda_Mo_Intro', 'i_believe_i_can_fly', 'discord_join_call', 'Phasmophobia_singing_Ghost', 'Huh5544', 'Dont_call_me_Needy_BFDI', 'scp_door_opening', 'G-toilet_lasers', 'Titanic_flute_fail', 'Wah_wah_wah_waaaaaaaahhh', 'Emotional_Damage', 'Screech_Car_Crash', 'Gta_v_notification', 'WHAT_ARE_YOU_AIMING_AT', 'Giornos_Theme_normal', 'PH_intro_x_See_you_again', 'Minecraft_-_Glass_Break', 'vekkars_ez4ence', 'Amazed_Emote_Animal_Crossing', 'I_fart_in_your_direction', 'Swoosh_Sound_Effects', 'Lionel_Richie_-_Hello_Is_It_Me', 'why_you_bully_me', 'Talk_Show_Intro', 'i_farted_and_a_poopy_almost_slipped_out', 'Lisa_AraAra', 'Sad_violin_14sec', 'What_meme_song', 'Quaaaaaaadra_kill', 'Crickets_Chirping', 'Deez_Nuts_Got_EM_AHAHAHAHA', 'Birdmaster86', 'Mission_failed_well_get_them', 'yorokobeshounen', 'Okay_Black_Guy_Vine', 'Mario_64_star_get_theme_song', 'Crying_Man', 'ehe_te_nandayo', 'Darth_Vader_NOOOOOOOOO', 'AKH', 'FinishHim', 'trollface_music', 'Death_sound_Fortnite', 'rip_bozo_minos_prime', 'Minecraft_Click', 'You_want_to_play_-_lets_play', 'Caught_a_Pokemon', 'Valorant_Defuse', 'Transformers_PSA_Jingle', 'mouse_click_by_ek6', 'Mission_Failed', 'brawl_stars_lose_slow', 'tom_and_jerry_opening', 'Valorant-Raze', 'Boom_Headshot', 'Limit_on_talking', 'Yoho_Happy_Birthday', 'Nervous_Gulp', 'YEETT', 'moan1', 'iconic_children_yay', 'Wind_noise', 'Very_Nice_Borat', 'Knuckles-Oh_No', 'risitas_laugh', 'Totally_Spies_Ringtone', 'im_in_danger', 'musica_triste_meme', 'Science_isnt_about_why.', 'Fart_Meme_Sound_Better_and_louder', '7', 'Le_fart_de_Simon', 'Bruh_meme', 'I_Dont_give_a_shita', 'Taco_Bell_Bong', 'Vuvuzela', 'Flappy_Birds_point', 'Bazinga', 'Dio_MudadaMUDAMUDAMUDAMUDA', 'Darius_in_the_closet', 'Discord_Jebaiting', 'check_mark', 'FEIN_FEIN_FEIN_FEIN', 'black_suit_spider_man', 'Rap_Battle_OOOHHHH', 'Target_Eliminated', 'fart_with_extra_reverb', 'RUN_vine', 'simp_over_girls_on_discord', 'Akhhhh...', 'Winamp_-_It_Really_Whips', 'Minecraft_Cave_Sound_18', 'noedolekciN', 'creaking_cupboard', 'Hey_listen', 'Glass_Breaking', 'Single_Heartbeat', 'Koyuki_Uwah', 'OOFITY_OOF', 'Discord_Ping', 'Stop_Youve_violated_the_law', 'Yes_King_AHHHHHHHHHHHHHHHH', 'WHAT_ARE_YOU_DOING_STEP_BRO', 'wait_a_minute_who_are_you_lite', 'n_a_n_i', 'Thug_life_instrumental', 'get_out', 'GULP_GULP_GULP', 'lego_breaking', 'Whip', 'aw_Shit_here_we_go_again', 'Why_are_we_still_here', 'bomb_has_been_defused', 'rev_up_those_fryers_-_shorter_version', 'BRUH', 'Bernard_Herrmann_-_Cape_Fear_theme', 'Smoke_Detector_Beep', 'Villager', 'Wow_Anime_meme', 'Its_a_DISASTER_by_TobiWAN', 'Ambatukam', 'Laser_Shot', 'Chamber_ult_VALORANT', 'I_Will_Always_Love_You', 'Indian_under_the_water', 'Oh_shit_Im_sorry', 'bitch_wtf', 'Aizen_Yokoso_Full', 'Hub_Intro_Sound', 'csgo_bomb_has_been_planted', 'Ora_Ora_Ora', 'Desert_Eagle_-_CS', 'Awkward_Moment', 'Door_Knock_2', 'Pokemon_Save_Game_Sound', 'iconic_damn', 'youtube_outro', 'outro_song', 'Pokemon-Level_Up', 'Directed_by_Robert_B_Weide', 'Crowd_Laughter_short', 'gay_central_cee', 'coffin_dance', 'AHHAHAHAHA', 'Explosion', 'Meme', 'Cinematic_Boom', 'Lightsaber_ON', 'spongebob_sad_song', 'VINE_BOOM_BASS_BOOSTED_MAN', 'Mario_Bros_Game_Over', 'Happy_happy_happy_song', 'Za_warudo_sound_effect', 'From_the_screen_to_the_ring', 'Combo_Breaker', 'Among_Us_role_reveal_sound', 'we_are_venom', 'Re.Zero_Whoaaayeeeaaayaaai', 'we_live_we_love_we_lie', 'Minecraft_Cave_Sound_19', 'FBI_open_UP', 'WTF_BOOM', 'xue_hua_piao_piao_bei_feng_xiao_xiao_meme', 'Doge_bonk', 'It_was_at_this_moment_he_knew', 'No_no_Wait_Wait', 'Noooo', 'Amogus_Full', 'Creaking_floor_door', 'IM_BOUT_TO_CUHHHH', 'Why_are_you_running', 'Ryujin_no_ken_wo_kurae', 'iPhone_Text', 'babi_kau_sajat', 'diamond_minecraft', 'Chasing_the_Sun_OOO', 'KING_CRIMSON', 'Law_and_Order_DUN_DUN', 'Objection', 'oof_minecraft', 'wake_up_meme', 'Coffin_Dance_Meme', 'packgod_loud', 'FF7_win', 'USSR_Anthem', 'frog_laughing_meme', 'Anime_Wow', 'Final_Fantasy_Victory_Fanfare', 'Kissing_Sound', 'Censor_Beep_2', 'Call_ambulance', 'Shinobu_Moshi_Moshi_Pseudo-ASMR', 'hog_rider', 'i_need_more_bullet', 'Teemo_laugh', 'you_are_my_special', 'Its_A_Me_Mario', 'Discord_Leave_Noise', 'oh_no_no_no_laugh', 'Scary_Tiktok_Music', 'mipan', 'Loud_Indian_music_short', 'ACK', 'Sitcom_Crowd_Ooh', 'Do_it_Palpa', 'minion_laughing', 'Pokmon_Black', 'Onee-sama', 'Ara_Ara', 'Ichigo_Bankai', 'Hello_Darkness_My_Old_Frieend', 'Star_trek_TNG_transporter', 'What_the_Hell_slv_soundss', 'NAKAKA_PUTANG_INA', 'OOOH_MY_GOD', 'Among_Us_Drip_Theme_Song', 'Sad_Musicccccc', 'Chipi_Chapa_meme', 'i_got_black_i_got_white_what_you_want', 'meme_violin_sad_violin', 'Hello_there-_obi_Wan', 'Zelda_-_Item_Get', 'Freddy_beatbox', 'Binky_sings_Happy_Birthday', 'NO_GOD_PLEASE_NO_NOOOOOOOO', 'ding_sound_effect', 'Michael_kenny_moan', 'superman_meme', 'Cute_Korean_Baby', 'ITS_A_DISASTER_6Mil_Echoslam', 'I_will_send_you_to_jesus', 'hitmarker', 'Belials_Okay', 'Mario_kart_start_race', 'Run_Meme', 'ta-da', 'Thick_Of_It_Brainrot', 'Sword_Cut', 'Minecraft_drinking_sound', 'doraemon_props', 'X_files', 'English_or_Spanish_Song', 'Mario_Yahoo', 'I_just_lost_my_dawg_original', 'Luigi_-_Woaaahh_Scream', 'hehe_boi_ainsley_harriott', 'OH_MY_GOD_Vine', 'how_you_doin..', 'Sakura_Kinomoto_-_HOEEEEEEEEE', 'To_be_Continued_jojo', 'oi_oi_baka', 'Fart_Button', 'Evil_Laugh', 'shut_up_patrick_stfu_up', 'Whatsapp_typing', 'samsung_notification', 'LOOK_AT_THIS_DUDE', 'I_will_send_you_to_Jesus_-_Steven_He', 'Smoke_Weed_EveryDay', 'AUUGHHH', 'spongebob_boowomp', 'valorant_footstep', 'Avengers_MCU_Melody', 'Aw_Shit_Here_go_again._CJ_from_GTA_SA', 'AHHHHH', 'Cat_scream_discordo', 'Call_of_the_Gigguk', 'Nico_Nico_nii', 'VINE_BOOM_SOUND', 'amogus_stuff', 'Pixel_Gun_3D_Purchase_Sound', 'Kamehameha', 'Xeno_Amazing', 'bonk_doge', 'illuminati_Confirmed', 'oiia_short', 'Talk_Show_Credits_Theme', 'Valorant_-_Chamber_Enemy_ULT', 'helicopter_helicopter_parakofer_parakofer', 'GET_OUT_Tuco', 'Luigi_Rolled.', 'kkon-doraemon-dougu', 'Yamate_Kudesai', 'BBQ', 'AUGHHHHH..._AUGHHHHH', 'kabuki_yo_sound_naruto', 'amogus_sussy', 'Guitar_meme', 'Fart_Song', 'sitcom_laugh', 'let_there_be_carnage', 'Alamak', 'mind', 'goofy_ahh_ohio_music', 'The_weather_outside_is_rizzy', 'Counter_Strike_-_go_go_go', 'metal_pipe_fall_meme', 'IVE_BEEN_WAITING_FOR_THIS', 'SPONGEBOB_2000_YEARS_LATER', 'Ngakak_laugh_annoying', 'Gawr_Gura_-_A', 'Diamlah_bodoh', 'atumalaca_hahahahaha', 'WRYYY', 'One_two_buckle_my_shoe', 'Mary_on_a_cross', 'gah_dayum', 'Door_Knocking_SOUND_EFFECT', 'Excellent', 'Rehehehe', 'wee_weee_weee', 'Creeper_Hiss', 'The_Entertainer_FUK', 'WHAT_ARE_YOU_DOING_IN_MY_SWAMP', 'TF2_-_Overtime', 'OHHHHHHH', 'Hello_im_under_the_water', 'Daijoubu', 'Hawk_Tuah', 'Emotional_Damage_Meme', 'Duck_toy_sound', 'Suara_Rem_Truk_Sumatra_Sulawesi_Kalimantan', 'FNAF_6_AM', 'Shika_Shika', 'RUN_vine_effect_sound', 'Bamboo_hit', 'Bye_have_a_great_time', 'Ka-Ching', 'MLG_PWNAGE', 'STOP_Button', 'ak47_loud', 'pikachu', 'Sr.Pelo_Boom', 'Balls_of_Steel', 'nope', 'Joseph_Joestar_Young_OH_NO', 'Random_fart_button', 'ReZero_Return_by_Death', 'Sexy_Sax', 'Huh_Ceeday', 'Mario_Jump', 'WHAT_THE_HELLLLLLLLLLLL', 'Mario_Farts', 'Squid_Game', 'Yurius_Skill_1', 'Metal_pipe_clang', 'super_idol', 'Round_One_Fight', 'Sike_Thats_The_Wrong_Number', 'Doors_Elevator_music', 'Was_that_the_bite_of_87', 'Lobotomy_Sound_Effect', 'vineboom', 'Throwing', 'Im_Batman', 'British_Quandale_Dingle', 'MLG_AIR_HORN', 'gen_alpha_bruuuu', 'Nokia_Arabic_Ringstone', 'Harder_Better_Faster_Whopper', 'THE_LOUDEST_NOISE_EVER', 'don_pollo_-_the_grefg', 'Angry_Korean_Gamer', 'Well_be_right_back', 'Bem_amigos_terminou', 'Ralp_Wiggum_Im_in_Danger', 'Supa_Hot_Fire', 'optimum_pride', 'Incorrect_Buzzer_Noise', 'Air_Horn_club_sample', 'Jazzy_Intro', 'Aris_Pan_Paka_Pan', 'Sukuna_laugh', 'Star_Platinum_Single_ORA', 'Nani_FULL', 'aughhhh_tiktok', 'Awkward_-_Crow_Silence', 'BADUM_TSS', 'punch_sound_effect_meme', 'the_weeknd_rizzz', 'VROOM_VROOM_IM_CAR_interminable_rooms', 'Correct_Answer_GameShow', 'toilet_sounds', 'Loud_Fart', 'sonic_ring', 'Teleport_sound', 'Pikmin', 'Roblox_Death_Sound_Effect', 'discord_call', 'Game_Show_Correct', 'GG', 'tawag_ng_tanghalan', 'Bongo_Run_SFX', 'Aww', 'Nani_what', 'Minecraft_LVL_up', 'Awkward_cricket', 'Windows_XP_-_Startup_Sound', 'BOMBASTIC_SIDE_EYE', 'Hells_kitchen_dramatic_sound', 'Flashback', 'Dramatic_Chipmunk', 'Yorokobe_Shounen', 'balloon_boy_hello_hi_fnaf', 'THATS_A_LOT_OF_DAMAGE', '4_Big_Guys.', 'revive_me_jett', 'John_Cena_ARE_YOU_SURE', '8-bit_Happy_Birthday', 'Fuwawa_hello', 'Mkay', 'I_need_more_bullets', 'Warcraft_Peon_-_Work_work', 'ZA_WARUDO_TIME_STOP', 'Shut_up_Meg', 'Sad_Ham', 'chinese_guy_rap', 'Woman_Scream_3', 'Street_Fighter_K.O', 'Grapefruit_technique', 'auf_der_heide', 'almendrero_de_doraemon', 'Elevator_Music_Background', 'Oniichan_i_have_a_dicc', 'Onii-chan_onii-chan', 'pokemon_battle', 'Goose_HONK', 'dun_dun_dunnnnnnnn', 'Darius_alarm', 'Cartoon_Mr._Krab_Walking', 'suprise_motherfer', 'Batle_Alarm_Star_Wars', 'Victory', 'panik_gak_masa_enggak', 'Marching_Soldiers', 'Trololo', 'Gas_Gas_Gas_-_Manuel_Short', 'Wow_Kongouratulations', 'Minecraft_Hit_Sound', 'WhatsApp_Bass_Boosted', 'Anime_punch', 'Squidward_Scream', 'MAN_SNORING_MEME', 'Boxing_Bell', 'Sad_Violin_the_meme_one', 'Stone_Sliding', 'Ghostly_sound', 'Its_Not_Just_A_Boulder_Its_A_Rock', 'Windows_3.1_startup_tada', 'The_Simpsons_-_Nelsons_HA-HA', 'Stop_it_Get_Some_Help', 'YEET', 'sus_clapping', 'Cat_laughing_at_you', 'JoJo_-_yes_yes_yes_yes_YES', 'Po_Pi_Po', 'Womp_Womp_Womp', 'execute_order_66', 'Im_fast_as_f_boi', 'Wait_a_minute_who_are_you', 'Loud_flash_bang', 'GunShotttt', 'bye_bye_mewing', 'Kids_Cheering_YAY', 'Leeroy_Jenkins', 'That_was_easy', 'Cartoon_run_take_off', 'Among_us_Roundstart', 'DUN_DUN', 'Honkai_Herta_Hudurin_kuru_kuru', 'Shooting_Stars', 'ROBLOX_oof', 'asian_meme_huh', 'NANI_SORE', 'that_one_josh_hutcherson_whistle_edit', 'Bongo_Feet', 'Eggman_Laugh_Sonic_Generations', 'Emergency_Paging_Dr_Beat', 'Ohio_ahh_sound_effect', 'What_the_Hell_Oh_My_Gawd_No_Way', 'correct_ding', 'What_the_hell_oh_ma_god_no_way_Anime', 'sad_meow_song', 'medal_clip_sound', 'CS-GO_Flashbang', 'RAZE_Fire_in_the_hole', 'Megumin_-_EXPLOSION', 'Minecraft_potion', 'Spiderman_2099_theme', 'sans_voice', 'omae_wa_mou_shindeiru_NANI', 'rizz_sound_effect', 'Freddy_fazbear_rizz', 'Windows_10_USB_connect', 'FART_1', 'Half_life_Hgrunt_-_MY._ASS._IS._HEAVY', 'Star_Platinum_Ora_Ora_Ora', 'Tik_Tok_India', 'Omae_wa_mou_shindeiru', 'Goofy_Yell', 'Minecraft_Explosions', 'Shame_-_Matt_Berry', 'What_-_Minion', 'skibidi_toilet', 'My_Name_Is_Jeff', 'JonTron_-_What_WTF', 'mongraall', 'Chipmunk_Laugh', 'Doom_Music', 'MR_BEAST_SCREAM', 'the_loudest_sound_in_the_universe', 'sword_draw_lol81849', 'Pokemon_Item_received', 'So_Close', 'Okay_lets_go', 'Screaming_Sheep', 'stop_posting_amogus', 'Super_Mario_Death', 'Uwah', 'clash_royale_hog_rider', 'YouTube_UWUUUUU', 'Touch_Jazzy_Part', 'BWAHAHAHA', 'fart_echo', 'George_Crying', 'Hell_Naw_Dog', 'BING_CHILLLLLLING', 'discordjoin', 'Gas_Gas_Gas_-_Manuel_Long', 'sova_no_where_to_run', 'anime_girl_singing_padoru', 'Knock_3D', 'bulli', 'jotaro_kujo_ora_ora_ora_jojo', 'ding_dong_eat_it_up', 'Galaxy_brain_meme', 'silence_i_kill_you', 'Shirakami_Fubuki_-_Yabe', 'Keyboard_Typing_Sound', 'Original_Sheesh', 'Buzzer', 'Ayaya_-_Oh_my_gah', 'minecraft_eating_sound', 'Houshou_Marine_-_Ahoy', 'spiderman_meme_song', 'Woy_Lagi_Santai_Kawan', 'Lego_Yoda_Death_Sound', 'GBF_-_Clarisse_Cheer', 'zab_zab_zab_bla_bla_blu_blu_alien_tiktok_meme', 'man_shut_yo_gah_damn_meme', 'Windows_XP_error_music', 'indian_song_7sek', 'cat_laugh_meme_1', 'and_his_name_is_John_Cenaaaaaa', 'Dolphin_Censor', 'Golden_freddy_laugh', 'Clink', '-999_Social_Credit_Siren', 'Hadouken', 'OH_NO_Jojo', 'my_mommy_said_no_more_skibidi_toilet', 'Siren_head', 'Venti_wah_waah_sound_Genshin_impact', 'i_i_i_be_poppin_bottles', 'running_in_the_90s', 'Long_brain_fart', 'The_long_and_winded_road_fart', 'Wrong_Answer_GameShow', 'OH_SHIT_echo', 'danger_alarm_sound_effect_meme', 'Lalalalala', 'Hello_darkness_my_old_friend', 'Water_Droplet_Drip', 'Rejoice', 'cartoon_snoring_sound_effect', 'Nanachi_bad', 'Saber_Alter_Singing_-_Fate_GO', 'Hello_There_Obi_Wan', 'Good_Bad_Ugly_Whistle', 'Ohio_xiao_mi_ringtone', 'Usada_Pekora_-_Sad_no...', 'Poi_Yuudachi', 'Confused_cross_eyed_kitten_meme', 'Aatrox_let_me_show_you_hell', 'loading..', 'Censor_Beep_3', 'Whos_That_Pokemon', 'Lightskin_Rizz_Sin_City', 'Rickroll_troll', 'psst_roblox_doors', 'Animal_Crossing_Isabelle_Voice', 'Goofy_ahh_car_horn_sound_effect', 'vine_boom_sound_effect_full', 'nope.avi', 'BYE_BYE', 'YEAH', 'sudden_suspense', 'SAMSUNG_NOTIFICATION_SOUND_EARRAPE', 'Shocked_sound', 'Ayaya', 'Jazz_music_stops', 'Crab_dance', 'Fart_Meme_Sound', 'Cartoon_slide_whistle', 'Door_Creak', 'Cute_UwU', 'fake_raze_ult_enemy', 'Wololo', 'discord_kitten', 'Yes_Mommy', 'Goofy_ahh_scream_by_vacooro', 'You_are_not_prepared', 'Gunshot_Play', 'Nintendo_Switch_Click', 'john_cena_chinese_meme', 'Succ', 'Fortnite_Default_Dance_Music', 'cartoon_poke', 'lack_of_a_father_figure', 'Bot', 'SPONGEBOB_ONE_HOUR_LATER', 'MAN_SMASHING_KEYBOARD', 'woosh', 'Whats_your_real_name', 'Airplane_Ding_Dong', 'photo', 'Resignation_Emote_Animal_Crossing', 'yoshi_tongue', 'Minecraft_Villager_Sound', 'ehehehhhh', 'MEEP_MERP', 'Gegagedigedagedago_EARRAPE', 'BRUHHH', 'Another_One_DJ_Khaled', 'Minecraft_TNT', 'Kai_Cenat_GYAAAAA-', 'target_lost...', 'Counter_Strike_-_Ok_lets_go', 'PE_KO_PE_KO_PE_KO_PE', 'Aww_Dang_It', 'Brooklyn_99', 'Spongebob_disappointed', 'Cartoon_Slip', 'Inception_Button', 'I_got_a_glock_in_my_Rari', 'Crowd_cheering', 'Clash_Royale_Startup', 'who_invited_this_kid', 'Mission_Impossible', 'oh_my_god_bro_oh_hell_nah_man', 'Minecraft_Horse_Death', 'fnaf_2_scream', 'why_are_you_gey', '', 'English_or_spanish', 'Hey_listen', 'hitmarker', 'quack', 'Crickets', 'ksi_new', 'totally_not_a_suspicious_button', 'ZA_WARUDO_TIME_STOP', '7', 'mind', 'OH_HELLO_THERE.', 'What_the_hell_oh_ma_god_no_way_Anime', 'BELLIGOL_BELLIGOL_BELLIGHAM', 'Pokmon_Black', 'BYE_BYE', 'Incorrect_Buzzer_Noise', 'PE_KO_PE_KO_PE_KO_PE', 'YEAH', 'Ryujin_no_ken_wo_kurae']
    
    # Perform classification on the provided audio file
    for pause in pauses:
        # Extract audio segment for the current pause
        pause_audio = extract_audio_segment(audio_file_path, sample_rate, pause['start'], pause['end'])
        
        # Pass the extracted audio segment to the audio classifier
        output = audio_classifier(pause_audio, candidate_labels=candidate_labels)
        
        # Sort the results based on the score in descending order
        sorted_output = sorted(output, key=lambda x: x['score'], reverse=True)
        
        # Get the top result
        top_result = sorted_output[0]
        
        # Check if the score is above 0.4
        if top_result['score'] > 0.4:
            top_label = top_result['label']
            top_score = top_result['score']
            results.append((top_label, top_score))
        else:
            results.append(None)
    
    return results


